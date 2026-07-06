import socket

def check_port(host, port):
    try:
        with socket.create_connection((host, port), timeout=5):
            print(f"Port {port} on {host} is open.")
            return True
    except Exception as e:
        print(f"Port {port} on {host} is blocked or unreachable. Error: {e}")
        return False

if __name__ == "__main__":
    smtp_host = "smtp.gmail.com"
    ports = [465, 587]

    for port in ports:
        check_port(smtp_host, port)
