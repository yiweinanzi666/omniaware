for p in 2422384 2420983; do echo PID=$p; cat /proc/$p/wchan 2>/dev/null || true; cat /proc/$p/syscall 2>/dev/null || true; ls -l /proc/$p/fd 2>/dev/null | tail -10; done
