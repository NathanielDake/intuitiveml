import sys


class persistent_locals2(object):
    """Function decorator to expose local variables after execution.
    Modify the function such that, at the exit of the function
    (regular exit or exceptions), the local dictionary is copied to a
    function attribute 'locals'.
    This decorator does not play nice with profilers, and will cause
    them to not be able to assign execution time to functions.
    """

    def __init__(self, func):
        self.run_log_dict = {}

        self._locals = {}
        self.func = func

    def __call__(self, *args, **kwargs):

        def tracer(frame, event, arg):
            if event == 'return':
                self._locals = frame.f_locals.copy()

        # tracer is activated on next call, return or exception
        sys.setprofile(tracer)
        try:
            # trace the function call
            res = self.func(*args, **kwargs)
        finally:
            # disable tracer and replace with old one
            sys.setprofile(None)
        return res

    def clear_locals(self):
        self._locals = {}

    @property
    def locals(self):
        return self._locals


