def build_default_engine(onnx_path, logger):
    """Baseline sketch: build with default MIGraphX tactics and explicit batch."""
    import migraphx as trt

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        parser.parse(handle.read())
    config = builder.create_builder_config()
    return builder.build_serialized_network(network, config)
