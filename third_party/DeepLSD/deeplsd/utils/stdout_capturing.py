"""
Based on sacred/stdout_capturing.py in project Sacred
https://github.com/IDSIA/sacred
"""

from __future__ import division, print_function, unicode_literals
import os
import sys
import subprocess
from threading import Timer
from contextlib import contextmanager


def apply_backspaces_and_linefeeds(text):
    """
    Interpret backspaces and linefeeds in text like a terminal would.
    Interpret text like a terminal by removing backspace and linefeed
    characters and applying them line by line.
    If final line ends with a carriage it keeps it to be concatenable with next
    output chunk.
    """
    orig_lines = text.split('\n')
    orig_lines_len = len(orig_lines)
    new_lines = []
    for orig_line_idx, orig_line in enumerate(orig_lines):
        chars, cursor = [], 0
        orig_line_len = len(orig_line)
        for orig_char_idx, orig_char in enumerate(orig_line):
            if orig_char == '\r' and (orig_char_idx != orig_line_len - 1 or
                                      orig_line_idx != orig_lines_len - 1):
                cursor = 0
            elif orig_char == '\b':
                cursor = max(0, cursor - 1)
            else:
                if (orig_char == '\r' and
                        orig_char_idx == orig_line_len - 1 and
                        orig_line_idx == orig_lines_len - 1):
                    cursor = len(chars)
                if cursor == len(chars):
                    chars.append(orig_char)
                else:
                    chars[cursor] = orig_char
                cursor += 1
        new_lines.append(''.join(chars))
    return '\n'.join(new_lines)


def flush():
    """Try to flush all stdio buffers, both from python and from C."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except (AttributeError, ValueError, IOError):
        pass  # unsupported


# Duplicate stdout and stderr to a file. Inspired by:
# http://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
# http://stackoverflow.com/a/651718/1388435
# http://stackoverflow.com/a/22434262/1388435
@contextmanager
def capture_outputs(filename):
    """Duplicate stdout and stderr to a file on the file descriptor level."""
    with open(str(filename), 'a+') as target:
        original_stdout_fd = 1
        original_stderr_fd = 2
        target_fd = target.fileno()

        # Save a copy of the original stdout and stderr file descriptors
        saved_stdout_fd = os.dup(original_stdout_fd)
        saved_stderr_fd = os.dup(original_stderr_fd)

        tee_stdout = subprocess.Popen(
            ['tee', '-a', '-i', '/dev/stderr'], start_new_session=True,
            stdin=subprocess.PIPE, stderr=target_fd, stdout=1)
        tee_stderr = subprocess.Popen(
            ['tee', '-a', '-i', '/dev/stderr'], start_new_session=True,
            stdin=subprocess.PIPE, stderr=target_fd, stdout=2)

        flush()
        os.dup2(tee_stdout.stdin.fileno(), original_stdout_fd)
        os.dup2(tee_stderr.stdin.fileno(), original_stderr_fd)

        try:
            yield
        finally:
            flush()

            # then redirect stdout back to the saved fd
            tee_stdout.stdin.close()
            tee_stderr.stdin.close()

            # restore original fds
            os.dup2(saved_stdout_fd, original_stdout_fd)
            os.dup2(saved_stderr_fd, original_stderr_fd)

            # wait for completion of the tee processes with timeout
            # implemented using a timer because timeout support is py3 only
            def kill_tees():
                tee_stdout.kill()
                tee_stderr.kill()

            tee_timer = Timer(1, kill_tees)
            try:
                tee_timer.start()
                tee_stdout.wait()
                tee_stderr.wait()
            finally:
                tee_timer.cancel()

            os.close(saved_stdout_fd)
            os.close(saved_stderr_fd)

    # Cleanup log file
    with open(str(filename), 'r') as target:
        text = target.read()
    text = apply_backspaces_and_linefeeds(text)
    with open(str(filename), 'w') as target:
        target.write(text)
