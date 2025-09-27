import subprocess, time

class ServerManager:
    def __init__(self, server_bin, model_path, chat_template, port=8080):
        self.server_bin = server_bin
        self.model_path = model_path
        self.chat_template = chat_template
        self.port = port
        self.proc = None

    def start(self):
        cmd=[
            self.server_bin,
            "-m", self.model_path,
            "-c", "8192",
            "-t", "6",
            "--n-gpu-layers", "40",
            "--temp", "0.001",
            "--top-p", "0.85",
            "--jinja",
            "--chat-template-file", self.chat_template,
            "--port", str(self.port)
        ]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"Started llama-server (PID {self.proc.pid} on port {self.port})")
        print(f"The chat template is {self.chat_template}")
        time.sleep(5) # Let the server warm up

    def stop(self):
        if self.proc:
            print(f"Stopping llama-server (PID {self.proc.pid})")
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                print("Forcing shutdown of llama-server")
                self.proc.kill()