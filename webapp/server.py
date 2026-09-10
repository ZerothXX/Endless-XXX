"""Loopback-only Flask UI with validated paths, uploads and serial GPU workers."""
import json
import os
from pathlib import Path
import queue
import re
import secrets
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError
from webapp.theme import CACHE, ROOT, fingerprint, mascot_assets
import config
from weight_selection import resolve_weight

RUNS = Path(config.LOGS_DIR) / "web_tasks"  # Task metadata only; no weights or generated images.
TOKEN = secrets.token_urlsafe(32)
NAME = re.compile(r"^[\w-]{1,48}$", re.UNICODE)


def character_name(value):
    if not isinstance(value, str) or not NAME.fullmatch(value) or value.upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1,10)], *[f"LPT{i}" for i in range(1,10)]}:
        raise ValueError("角色名只能包含文字、数字、下划线和连字符，最长 48 个字符")
    return value


def characters():
    found = {}
    for folder in (ROOT / "dataset").iterdir():
        if folder.is_dir() and (folder / "images").is_dir() and NAME.fullmatch(folder.name):
            try:
                weight = resolve_weight(folder.name, allow_missing=True)
                if weight:
                    found[folder.name] = weight
            except ValueError:
                continue
    return found


class Jobs:
    def __init__(self):
        self.items, self.processes = {}, {}
        self.lock, self.queue = threading.RLock(), queue.Queue()
        RUNS.mkdir(parents=True, exist_ok=True)
        for path in RUNS.glob("*/job.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["state"] in {"running", "queued"}:
                    item.update(state="error", phase="服务已重启，请重新开始任务")
                    path.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
                self.items[item["id"]] = item
            except (ValueError, KeyError):
                continue
        threading.Thread(target=self.work, daemon=True).start()

    def save(self, item):
        path = RUNS / item["id"] / "job.json"
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)

    def submit(self, spec):
        with self.lock:
            job_id = f"{int(time.time())}_{secrets.token_hex(5)}"
            target = RUNS / job_id
            target.mkdir()
            spec["directory"] = str(target)
            (target / "spec.json").write_text(json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            item = {"id": job_id, "kind": spec["kind"], "character": spec["character"],
                    "state": "queued", "phase": "等待 GPU 空闲", "percent": 0, "log_tail": []}
            self.items[job_id] = item
            self.save(item)
            self.queue.put(job_id)
            return job_id

    def pause(self, job_id):
        with self.lock:
            item = self.items[job_id]
            if item["state"] in {"queued", "running"}:
                item.update(state="paused", phase="已暂停，可从当前输入重新开始")
                proc = self.processes.get(job_id)
                if proc and proc.poll() is None:
                    proc.terminate()
                self.save(item)

    def work(self):
        while True:
            job_id = self.queue.get()
            item = self.items[job_id]
            target = RUNS / job_id
            try:
                with self.lock:
                    if item["state"] == "paused":
                        continue
                    item.update(state="running", phase="启动任务")
                    self.save(item)
                    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                    proc = subprocess.Popen([sys.executable, "-u", str(ROOT / "webapp/worker.py"), str(target / "spec.json")],
                            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", env=env,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                    self.processes[job_id] = proc
                with (target / "run.log").open("w", encoding="utf-8") as log:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        with self.lock:
                            if item["state"] == "paused":
                                continue
                            if line.startswith("@EVENT "):
                                event = json.loads(line[7:])
                                item["percent"] = max(item["percent"], min(99, event.pop("percent", 0)))
                                item.update(event)
                            elif line.strip():
                                item["log_tail"] = (item["log_tail"] + [line.strip()])[-100:]
                            self.save(item)
                code = proc.wait()
                with self.lock:
                    if item["state"] != "paused":
                        item["state"] = "done" if code == 0 and "result" in item else "error"
                        if item["state"] == "done":
                            item["percent"] = 100
                        else:
                            item["phase"] = "任务失败，请查看日志"
                        self.save(item)
                    if item["state"] == "done" and item["kind"] == "train" and item["result"].get("precompute_theme"):
                        # A separate process after training has exited: no concurrent GPU models.
                        self.submit({"kind": "theme", "character": item["character"], "weight": item["result"]["weight"]})
            except Exception as exc:
                with self.lock:
                    item.update(state="error", phase=str(exc))
                    self.save(item)
            finally:
                self.processes.pop(job_id, None)
                self.queue.task_done()


def create_app(start_jobs=True):
    app = Flask(__name__, static_folder="static", static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    jobs = Jobs() if start_jobs else None
    app.extensions["jobs"] = jobs

    @app.before_request
    def protect():
        if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
            abort(403)
        if request.method == "POST":
            if request.headers.get("X-Local-Token") != TOKEN:
                abort(403)
            origin = request.headers.get("Origin")
            if origin and urlsplit(origin).netloc != request.host:
                abort(403)

    @app.errorhandler(ValueError)
    def invalid(exc):
        return jsonify(error=str(exc)), 400

    @app.get("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    @app.get("/api/bootstrap")
    def bootstrap():
        return jsonify(token=TOKEN)

    @app.get("/api/characters")
    def choices():
        return jsonify([{"name": name, "has_model": True} for name in characters()])

    @app.get("/api/theme/<name>")
    def theme_cache(name):
        name = character_name(name)
        weight = characters().get(name)
        if not weight:
            abort(404)
        stamp = fingerprint(ROOT / "dataset" / name, weight)
        path = CACHE / name / stamp / "theme.json"
        return jsonify(json.loads(path.read_text(encoding="utf-8"))) if path.exists() else (jsonify(cached=False), 404)

    @app.post("/api/theme/<name>")
    def theme_build(name):
        name = character_name(name)
        weight = characters().get(name)
        if not weight:
            raise ValueError("角色尚无完成的权重")
        with jobs.lock:
            existing = next((j["id"] for j in jobs.items.values() if j["kind"] == "theme" and j["character"] == name
                             and j["state"] in {"running", "queued"}), None)
            return jsonify(job_id=existing or jobs.submit({"kind": "theme", "character": name, "weight": weight}))

    def uploads(training=False):
        files = request.files.getlist("images")
        if not 1 <= len(files) <= 20:
            raise ValueError("请选择 1 至 20 张图片")
        decoded = []
        seen = set()
        total_pixels = 0
        for i, file in enumerate(files):
            try:
                image = ImageOps.exif_transpose(Image.open(file.stream))
                if image.width * image.height > 30_000_000:
                    raise ValueError("单张图片不得超过 3000 万像素")
                total_pixels += image.width * image.height
                if total_pixels > 80_000_000:
                    raise ValueError("单批图片总像素过大，请缩小参考图后重试")
                image.load()
                filename = Path(file.filename.replace("\\", "/")).name
                stem = Path(filename).stem
                if not NAME.fullmatch(stem) or stem in seen:
                    stem = f"image_{i + 1:03d}"
                while stem in seen:
                    stem += "_"
                seen.add(stem)
                decoded.append((stem + ".png", image.convert("RGBA")))
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                raise ValueError("上传内容不是有效图片")
        return decoded

    @app.post("/api/generate")
    def generate():
        name = character_name(request.form.get("character"))
        weight = characters().get(name)
        if not weight:
            raise ValueError("需先选择已训练角色")
        decoded = uploads()
        target = ROOT / "input/web_workspace" / secrets.token_hex(12)
        target.mkdir(parents=True)
        paths = []
        for filename, image in decoded:
            image.save(target / filename)
            paths.append(str(target / filename))
        return jsonify(job_id=jobs.submit({"kind": "generate", "character": name, "weight": weight, "images": paths}))

    @app.post("/api/train")
    def train():
        name = character_name(request.form.get("name"))
        decoded = uploads(True)
        steps, resolution = int(request.form.get("steps", 600)), int(request.form.get("resolution", 640))
        if not 50 <= steps <= 10000 or resolution not in {512, 640, 768}:
            raise ValueError("训练参数超出范围")
        caption = request.files.get("captions")
        content = None
        if caption:
            if caption.filename != "captions.txt":
                raise ValueError("标签文件必须命名为 captions.txt")
            content = caption.read(1024 * 1024 + 1)
            if len(content) > 1024 * 1024:
                raise ValueError("标签文件过大")
            try:
                content = content.decode("utf-8-sig")
            except UnicodeDecodeError:
                raise ValueError("标签文件需要 UTF-8 编码")
            # Image bytes are normalized to PNG; preserve stems and remap caption suffixes.
            content = re.sub(r"(?m)^([^,]+)\.(?:jpg|jpeg|webp)(?=,)", r"\1.png", content, flags=re.I)
        target = ROOT / "dataset" / name
        try:
            target.mkdir()
        except FileExistsError:
            raise ValueError("角色目录已存在，不会覆盖。请使用新的角色名")
        (target / "images").mkdir()
        for filename, image in decoded:
            image.save(target / "images" / filename)
        if content is not None:
            (target / "captions.txt").write_text(content, encoding="utf-8")
        return jsonify(job_id=jobs.submit({"kind": "train", "character": name, "steps": steps,
                    "resolution": resolution, "precompute_theme": request.form.get("precompute_theme") == "true"}))

    @app.get("/api/jobs/<job_id>")
    def status(job_id):
        with jobs.lock:
            if job_id not in jobs.items:
                abort(404)
            return jsonify(jobs.items[job_id])

    @app.post("/api/jobs/<job_id>/pause")
    def pause(job_id):
        if job_id not in jobs.items:
            abort(404)
        jobs.pause(job_id)
        return jsonify(state="paused")

    @app.post("/api/jobs/<job_id>/restart")
    def restart(job_id):
        with jobs.lock:
            item = jobs.items.get(job_id)
            if not item or item["state"] not in {"paused", "error"}:
                raise ValueError("只能重新开始已暂停或失败的任务")
            spec = json.loads((RUNS / job_id / "spec.json").read_text(encoding="utf-8"))
            return jsonify(job_id=jobs.submit(spec))

    @app.post("/api/files/result")
    def open_result():
        path = Path(config.RESULT_DIR)
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))
        return jsonify(path=str(path))

    @app.get("/themes/<path:filename>")
    def theme_asset(filename):
        if Path(filename).suffix.lower() != ".png":
            abort(404)
        return send_from_directory(CACHE, filename)

    @app.get("/outputs/<kind>/<path:filename>")
    def output(kind, filename):
        directory = {"result": config.RESULT_DIR, "curves": config.CURVES_DIR}.get(kind)
        if directory is None:
            abort(404)
        if Path(filename).suffix.lower() != ".png":
            abort(404)
        return send_from_directory(directory, filename)

    return app


def main():
    mascot_assets()
    create_app().run(host="127.0.0.1", port=7860, debug=False, threaded=True)
