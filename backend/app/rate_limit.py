from slowapi import Limiter
from slowapi.util import get_remote_address

# In-memory, per-IP limiter. Fine for a single-instance deployment like the
# free Render demo; a multi-instance deployment would need a shared backend
# (e.g. Redis) since each process would otherwise track its own counts.
limiter = Limiter(key_func=get_remote_address)
