import tensorrt as trt

class LayerProfiler(trt.IProfiler):
    def __init__(self):
        super(LayerProfiler, self).__init__()

    def report_layer_time(self, layer_name, ms):
        # 这里可以通过层名称记录信息或做进一步操作
        print(f"Layer {layer_name} took {ms} ms to execute")