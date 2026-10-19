"""仅监听本机的角色工作台。"""
import atexit
import json
import mimetypes
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from urllib.parse import urlsplit

from flask import Flask, abort, jsonify, request, send_from_directory
from PIL import Image, ImageOps, UnidentifiedImageError

import config
import dataset
from character import semantics

ROOT = Path(config.PROJECT_ROOT)
WEB = ROOT / "web"
DATASET = Path(config.DATASET_DIR)
PACKAGES = Path(config.CHARACTERS_DIR)
RESULTS = Path(config.RESULT_DIR)
STATE = Path(config.OUTPUT_DIR) / "web"
RUNS = STATE / "jobs"
TOKEN = secrets.token_urlsafe(32)
ENTRY_LOCK = threading.RLock()
NAME = re.compile(r"^[\w-]{1,48}$", re.UNICODE)
IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}
mimetypes.add_type("text/javascript", ".js")


def character_name(value):
    reserved = {"CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10))}
    if (not isinstance(value, str) or not NAME.fullmatch(value)
            or value.upper() in reserved or value.endswith((".", " "))):
        raise ValueError("角色名只能包含文字、数字、下划线和连字符，最长 48 个字符")
    return value


def natural_key(value):
    return [int(part) if part.isdigit() else part.casefold()
            for part in re.split(r"(\d+)", value)]


def role_path(name):
    return DATASET / character_name(name)


def package_path(name):
    return PACKAGES / character_name(name)


def roles():
    names = {p.name for base in (DATASET, PACKAGES) if base.is_dir()
             for p in base.iterdir() if p.is_dir() and NAME.fullmatch(p.name)}
    result = []
    for name in sorted(names, key=natural_key):
        try:
            ready = bool(semantics.audit_switch(name).get("ready"))
        except (OSError, ValueError, KeyError):
            ready = False
        result.append({"name": name, "trained": ready,
                       "dataset": role_path(name).is_dir()})
    return result


def read_hidden():
    path = STATE / "hidden.json"
    try:
        return set(json.loads(path.read_text(encoding="utf-8")))
    except (FileNotFoundError, ValueError, TypeError):
        return set()


def write_hidden(names):
    STATE.mkdir(parents=True, exist_ok=True)
    path = STATE / "hidden.json"
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(sorted(names, key=natural_key),
                               ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def decode_images(files, limit=20):
    if len(files) > limit:
        raise ValueError(f"每次最多添加 {limit} 张图片")
    images = []
    pixels = 0
    for file in files:
        if Path(file.filename or "").suffix.lower() not in IMAGE_EXTS:
            raise ValueError("只支持 PNG、JPG、WEBP 图片")
        try:
            with Image.open(file.stream) as source:
                image = ImageOps.exif_transpose(source)
                pixels += image.width * image.height
                if image.width * image.height > 30_000_000 or pixels > 80_000_000:
                    raise ValueError("图片尺寸过大")
                image.load()
                images.append(image.convert("RGB"))
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
            raise ValueError("上传内容不是有效图片")
    return images


def save_images(images, target, prefix="image"):
    target.mkdir(parents=True, exist_ok=True)
    names = []
    try:
        for image in images:
            filename = f"{prefix}_{int(time.time() * 1000)}_{secrets.token_hex(4)}.png"
            names.append(filename)
            image.save(target / filename)
    except Exception:
        for filename in names:
            (target / filename).unlink(missing_ok=True)
        raise
    return names


def cleanup_uploads(spec):
    """删除已结束任务的临时输入，严格限制在本工作台的上传目录内。"""
    if spec.get("kind") != "generate" or not spec.get("images"):
        return
    base = (STATE / "uploads").resolve()
    paths = [Path(path).resolve() for path in spec["images"]]
    parent = paths[0].parent
    if parent.parent == base and all(path.parent == parent for path in paths):
        shutil.rmtree(parent, ignore_errors=True)


class Jobs:
    def __init__(self):
        self.items = {}
        self.processes = {}
        self.lock = threading.RLock()
        self.queue = queue.Queue()
        RUNS.mkdir(parents=True, exist_ok=True)
        for path in RUNS.glob("*/job.json"):
            try:
                item = json.loads(path.read_text(encoding="utf-8"))
                if item["state"] in {"running", "queued"}:
                    item.update(state="error", phase="服务已重启，请重新开始")
                    self.save(item)
                self.items[item["id"]] = item
            except (OSError, ValueError, KeyError):
                continue
        threading.Thread(target=self.work, daemon=True).start()
        atexit.register(self.shutdown)

    def shutdown(self):
        """正常退出服务时停止工作进程，避免重启后遗留 GPU 任务。"""
        with self.lock:
            processes = list(self.processes.values())
            for item in self.items.values():
                if item["state"] in {"running", "queued"}:
                    item.update(state="error", phase="服务已关闭，请重新开始")
                    self.save(item)
        for proc in processes:
            if proc.poll() is None:
                try:
                    proc.terminate()
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                except OSError:
                    pass

    def save(self, item):
        target = RUNS / item["id"] / "job.json"
        temp = target.with_suffix(".tmp")
        temp.write_text(json.dumps(item, ensure_ascii=False), encoding="utf-8")
        temp.replace(target)

    def submit(self, spec):
        with self.lock:
            for item in self.items.values():
                if (item["character"] == spec["character"]
                        and item["state"] in {"running", "queued"}):
                    raise ValueError("该角色已有任务正在进行")
            job_id = f"{int(time.time())}_{secrets.token_hex(5)}"
            path = RUNS / job_id
            path.mkdir(parents=True)
            (path / "spec.json").write_text(
                json.dumps(spec, ensure_ascii=False), encoding="utf-8")
            item = {"id": job_id, "kind": spec["kind"],
                    "character": spec["character"], "state": "queued",
                    "phase": "等待任务开始", "percent": 0, "log_tail": [],
                    "created_at": time.time_ns()}
            if spec["kind"] == "generate":
                item["total"] = len(spec["images"])
            self.items[job_id] = item
            self.save(item)
            self.queue.put(job_id)
            return job_id

    def pause(self, job_id):
        with self.lock:
            item = self.items.get(job_id)
            if item is None:
                abort(404)
            if item["state"] in {"running", "queued"}:
                item.update(state="paused", phase="训练暂停" if item["kind"] == "train"
                            else "转化已停止")
                proc = self.processes.get(job_id)
                if proc and proc.poll() is None:
                    proc.terminate()
                if item["kind"] == "generate":
                    spec = json.loads((RUNS / job_id / "spec.json").read_text(
                        encoding="utf-8"))
                    name = item["character"]
                    folder = RESULTS / f"{name}_result"
                    files = [folder / f"{name}_{Path(path).stem}_result.png"
                             for path in spec["images"]]
                    item["result"] = {"images": [
                        f"/api/results/{name}/{path.name}"
                        for path in files if path.is_file()]}
                self.save(item)
            return item

    def work(self):
        while True:
            job_id = self.queue.get()
            item = self.items[job_id]
            target = RUNS / job_id
            proc = None
            spec = None
            try:
                with self.lock:
                    if item["state"] != "queued":
                        continue
                    item.update(state="running", phase="正在启动")
                    self.save(item)
                    env = dict(os.environ, PYTHONIOENCODING="utf-8",
                               PYTHONUNBUFFERED="1")
                    proc = subprocess.Popen(
                        [sys.executable, "-u", str(WEB / "worker.py"),
                         str(target / "spec.json")],
                        cwd=str(ROOT), stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                        errors="replace", env=env,
                        creationflags=(subprocess.CREATE_NO_WINDOW
                                       if os.name == "nt" else 0))
                    self.processes[job_id] = proc
                with (target / "run.log").open("w", encoding="utf-8") as log:
                    for line in proc.stdout:
                        log.write(line)
                        log.flush()
                        with self.lock:
                            if item["state"] == "paused":
                                continue
                            if line.startswith("@EVENT "):
                                try:
                                    event = json.loads(line[7:])
                                except ValueError:
                                    item["log_tail"] = (item["log_tail"]
                                                        + [line.rstrip()])[-500:]
                                else:
                                    item["percent"] = min(100, event.pop("percent", 0))
                                    item.update(event)
                            elif line.strip():
                                item["log_tail"] = (item["log_tail"]
                                                    + [line.rstrip()])[-500:]
                                if item["kind"] == "generate":
                                    match = re.search(r"\[test\] 已保存: (.+)", line)
                                    if match:
                                        item["phase"] = "图片转化中……"
                            self.save(item)
                code = proc.wait()
                with self.lock:
                    if item["state"] == "running":
                        item["state"] = ("done" if code == 0 and "result" in item
                                         else "error")
                        if item["state"] == "error":
                            item["phase"] = "任务失败"
                        self.save(item)
            except Exception as exc:
                if proc and proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
                with self.lock:
                    if item["state"] != "paused":
                        item.update(state="error", phase=str(exc))
                        self.save(item)
            finally:
                try:
                    spec = json.loads((target / "spec.json").read_text(encoding="utf-8"))
                    cleanup_uploads(spec)
                except (OSError, ValueError, TypeError):
                    pass
                with self.lock:
                    self.processes.pop(job_id, None)
                self.queue.task_done()


def create_app(start_jobs=True):
    app = Flask(__name__, static_folder=str(WEB / "static"),
                static_url_path="/static")
    app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
    jobs = Jobs() if start_jobs else None
    app.extensions["jobs"] = jobs

    @app.before_request
    def protect():
        if request.host.split(":")[0] not in {"127.0.0.1", "localhost"}:
            abort(403)
        if request.method in {"POST", "DELETE"}:
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
    def characters():
        hidden = read_hidden()
        return jsonify([{**role, "hidden": role["name"] in hidden}
                        for role in roles()])

    @app.post("/api/characters")
    def create_character():
        name = character_name((request.get_json() or {}).get("name"))
        if role_path(name).exists() or package_path(name).exists():
            raise ValueError("已有该角色，如需更改请前往重训角色")
        (role_path(name) / "images").mkdir(parents=True)
        return jsonify(name=name), 201

    @app.post("/api/characters/<name>/entries")
    def add_entries(name):
        name = character_name(name)
        target = role_path(name)
        if not target.is_dir():
            abort(404)
        mode = request.form.get("mode", "character")
        if mode not in {"character", "mark"}:
            raise ValueError("未知输入类型")
        description = request.form.get("description", "").strip()
        if len(description) > 4000:
            raise ValueError("描述最多 4000 字")
        description = " ".join(description.split())
        files = request.files.getlist("images")
        if mode == "mark" and (len(files) != 1 or not description):
            raise ValueError("标志图像一次只能上传一张，且必须填写描述")
        if not files and not description:
            raise ValueError("请输入描述或添加图片")
        images = decode_images(files)
        with ENTRY_LOCK:
            names = save_images(images, target / "images",
                                prefix="_mark" if mode == "mark" else "image")
            try:
                if mode == "mark":
                    with (target / "mark_refs.txt").open("a", encoding="utf-8") as stream:
                        stream.write(f"images/{names[0]}\t{description}\n")
                elif description:
                    with (target / "marks.txt").open("a", encoding="utf-8") as stream:
                        stream.write(description + "\n")
            except Exception:
                for filename in names:
                    (target / "images" / filename).unlink(missing_ok=True)
                raise
        return jsonify(images=[f"/api/dataset/{name}/{file}" for file in names],
                       description=description, mode=mode)

    @app.get("/api/dataset/<name>/<filename>")
    def dataset_image(name, filename):
        character_name(name)
        if Path(filename).name != filename or not filename.endswith(".png"):
            abort(404)
        return send_from_directory(role_path(name) / "images", filename)

    @app.get("/api/characters/<name>/entries")
    def entries(name):
        target = role_path(name)
        if not target.is_dir():
            abort(404)
        pictures = sorted((target / "images").glob("*.png"),
                          key=lambda path: path.stat().st_mtime)
        return jsonify(images=[f"/api/dataset/{name}/{p.name}" for p in pictures])

    @app.post("/api/characters/<name>/visibility")
    def visibility(name):
        character_name(name)
        if name not in {role["name"] for role in roles()}:
            abort(404)
        hidden = read_hidden()
        if (request.get_json() or {}).get("hidden"):
            hidden.add(name)
        else:
            hidden.discard(name)
        write_hidden(hidden)
        return jsonify(hidden=name in hidden)

    @app.delete("/api/characters/<name>")
    def delete_character(name):
        character_name(name)
        target = package_path(name).resolve()
        parent = PACKAGES.resolve()
        if target.parent != parent or not target.is_dir():
            abort(404)
        if (request.get_json() or {}).get("confirm") != name:
            raise ValueError("请确认要删除的角色")
        if jobs and any(item["character"] == name and item["state"] in
                        {"queued", "running"} for item in jobs.items.values()):
            raise ValueError("该角色有任务正在进行")
        shutil.rmtree(target)
        hidden = read_hidden()
        hidden.discard(name)
        write_hidden(hidden)
        return jsonify(deleted=name)

    @app.post("/api/generate")
    def generate():
        name = character_name(request.form.get("character"))
        if not semantics.audit_switch(name).get("ready"):
            raise ValueError("需先选择已训练角色")
        images = decode_images(request.files.getlist("images"))
        if not images:
            raise ValueError("请输入图片")
        target = STATE / "uploads" / secrets.token_hex(12)
        try:
            names = save_images(images, target)
            paths = [str(target / file) for file in names]
            job_id = jobs.submit(
                {"kind": "generate", "character": name, "images": paths})
        except Exception:
            shutil.rmtree(target, ignore_errors=True)
            raise
        return jsonify(job_id=job_id)

    @app.post("/api/train")
    def train():
        name = character_name((request.get_json() or {}).get("name"))
        if not role_path(name).is_dir():
            abort(404)
        try:
            validation = dataset.validate_dataset(name)
        except (FileNotFoundError, RuntimeError):
            raise ValueError("图片不足10张")
        if not validation["ok"]:
            raise ValueError("图片不足10张")
        return jsonify(job_id=jobs.submit({"kind": "train", "character": name}))

    @app.get("/api/jobs/<job_id>")
    def job(job_id):
        if not jobs or job_id not in jobs.items:
            abort(404)
        with jobs.lock:
            return jsonify(jobs.items[job_id])

    @app.get("/api/jobs/latest/<name>/<kind>")
    def latest_job(name, kind):
        character_name(name)
        if kind not in {"train", "generate"} or not jobs:
            abort(404)
        with jobs.lock:
            matches = [item for item in jobs.items.values()
                       if item["character"] == name and item["kind"] == kind]
            return jsonify(job=max(matches, key=lambda item: item.get("created_at", 0))
                           if matches else None)

    @app.post("/api/jobs/<job_id>/pause")
    def pause(job_id):
        if not jobs:
            abort(404)
        return jsonify(jobs.pause(job_id))

    @app.post("/api/open-folder")
    def open_folder():
        data = request.get_json() or {}
        name = character_name(data.get("name"))
        kind = data.get("kind")
        if kind == "result" and semantics.audit_switch(name).get("ready"):
            path = RESULTS / f"{name}_result"
        elif kind == "dataset" and role_path(name).is_dir():
            path = role_path(name)
        else:
            abort(404)
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(path))
        return jsonify(path=str(path))

    @app.get("/api/results/<name>/<filename>")
    def result_image(name, filename):
        character_name(name)
        if (Path(filename).name != filename
                or not filename.endswith("_result.png")):
            abort(404)
        return send_from_directory(RESULTS / f"{name}_result", filename)

    return app


def main():
    create_app().run(host="127.0.0.1", port=7860, debug=False, threaded=True)


if __name__ == "__main__":
    main()
