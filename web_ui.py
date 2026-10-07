"""本地角色工作台入口（新前端 web/）。

用项目的 DL Python 环境运行；仅监听本机 http://127.0.0.1:7860

    python web_ui.py

前端为 web/static/（index.html / style.css / app.js），服务端为 web/server.py，
GPU 任务由 web/worker.py 在独立子进程里调用项目正式入口 train.py / test.py 执行。
"""
from web.server import main

if __name__ == "__main__":
    main()
