# gunicorn.conf.py

# Bind to the host and port expected by Render
bind = "0.0.0.0:10000"  # change port if Render specifies a different one

# Workers & threads
workers = 1              # only one worker to save memory
threads = 1              # single thread per worker

# Timeout settings
timeout = 120            # allow up to 2 minutes per request

# Preload app to reduce memory usage
preload_app = True

# Logging
accesslog = "-"          # logs to stdout (Render shows stdout logs)
errorlog = "-"           # logs to stdout
loglevel = "info"
