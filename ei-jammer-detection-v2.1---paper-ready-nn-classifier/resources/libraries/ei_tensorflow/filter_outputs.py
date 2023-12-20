import io, os, sys, ctypes, tempfile, typing
from contextlib import contextmanager

libc = ctypes.CDLL(None)

to_filter = [
    'Estimated count of arithmetic ops',
    'fully_quantize:'
]

# Prints lines from a StringIO stream, but omits any that start with a string the to_filter list
def print_filtered_output(output: io.StringIO):
    value = output.getvalue()
    for line in iter(value.splitlines()):
        caught = False
        for filter in to_filter:
            if line.startswith(filter):
                caught = True
                break
        if not caught:
            print(line)

@contextmanager
# Redirects either stderr or stdout to a stream, including from C. The built in redirect_stdout does not
# handle the output of C modules.
# Inspired by this https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python
def output_redirector(mode: typing.Literal['stdout', 'stderr'], stream: io.StringIO):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    if mode == 'stdout':
        original_output_fd = sys.stdout.fileno()
        c_output = ctypes.c_void_p.in_dll(libc, 'stdout')
    else:
        original_output_fd = sys.stderr.fileno()
        c_output = ctypes.c_void_p.in_dll(libc, 'stderr')

    def _redirect_output(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_output)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        if mode == 'stdout':
            sys.stdout.close()
        else:
            sys.stderr.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_output_fd)
        # Create a new sys.stdout that points to the redirected fd
        if mode == 'stdout':
            sys.stdout = io.TextIOWrapper(os.fdopen(original_output_fd, 'wb'))
        else:
            sys.stderr = io.TextIOWrapper(os.fdopen(original_output_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_output_fd = os.dup(original_output_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_output(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_output(saved_output_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read().decode('utf-8'))
    finally:
        tfile.close()
        os.close(saved_output_fd)
