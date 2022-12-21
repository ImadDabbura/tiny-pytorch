class Op:
    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args):
        raise NotImplementedError()

    def gradient(self, out_grad, out_node):
        raise NotImplementedError()
