import os
import tensorrt as trt


ONNX_SIM_MODEL_PATH = 'weights/grounded_opset17_fixed_4.onnx'
TENSORRT_ENGINE_PATH_PY = 'weights/grounded_opset17_fixed_fp16_4.engine'


def build_engine(onnx_file_path, engine_file_path, flop=16):
    trt_logger = trt.Logger(trt.Logger.VERBOSE)  # 记录详细的模型转换日志（包括每一层的详细信息）
    builder = trt.Builder(trt_logger)
    network = builder.create_network(
        1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )

    # 解析ONNX模型
    parser = trt.OnnxParser(network, trt_logger)
    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            print('ERROR: Failed to parse the ONNX file.')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None
    print("Completed parsing ONNX file")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)

    # 如果FP16支持则启用
    if builder.platform_has_fast_fp16 and flop == 16:
        print("Export FP16 engine")
        config.set_flag(trt.BuilderFlag.FP16)

    if os.path.isfile(engine_file_path):
        try:
            os.remove(engine_file_path)
        except Exception:
            print("Cannot remove existing file: ", engine_file_path)

    print("Creating TensorRT Engine")

    config.set_tactic_sources(1 << int(trt.TacticSource.CUBLAS))

    profile = builder.create_optimization_profile()

    # 设置输入维度（固定）
    profile.set_shape("img", (1, 3, 800, 1200), (1, 3, 800, 1200), (1, 3, 800, 1200))  # Example shapes for 'img'
    profile.set_shape("input_ids", (1, 4), (1, 4), (1, 4))  # Adjust the sequence lengths accordingly
    profile.set_shape("attention_mask", (1, 4), (1, 4), (1, 4))
    profile.set_shape("position_ids", (1, 4), (1, 4), (1, 4))
    profile.set_shape("token_type_ids", (1, 4), (1, 4), (1, 4))
    profile.set_shape("text_token_mask", (1, 4, 4), (1, 4, 4), (1, 4, 4))

    config.add_optimization_profile(profile)

    # 编译序列化engine文件
    serialized_engine = builder.build_serialized_network(network, config)

    # 打印出分析结果
    # inspector.print_layer_times()

    # 如果引擎创建失败
    if serialized_engine is None:
        print("引擎创建失败")
        return None

    # 将序列化的引擎保存到文件
    with open(engine_file_path, "wb") as f:
        f.write(serialized_engine)
    print("序列化的引擎已保存到: ", engine_file_path)

    return serialized_engine


if __name__ == "__main__":
    build_engine(ONNX_SIM_MODEL_PATH, TENSORRT_ENGINE_PATH_PY)
