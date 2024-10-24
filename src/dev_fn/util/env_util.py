import os


def read_environ(pid):
    environ_path = f"/proc/{pid}/environ"
    env_vars = {}
    try:
        with open(environ_path, 'r') as file:
            # The variables in this file are null-byte (\0) delimited
            environ_content = file.read()
            for env_entry in environ_content.split('\0'):
                if '=' in env_entry:
                    key, value = env_entry.split('=', 1)
                    env_vars[key] = value
    except Exception as e:
        pass
    return env_vars


def read_shell_environ():
    return read_environ(os.getppid())
