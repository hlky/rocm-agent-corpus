def build_profiled_engine(onnx_path, logger, timing_cache_path=None, fp16=True):
    """Optimized sketch: make tactic, profile, precision, and timing-cache choices explicit."""
    import migraphx as trt

    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as handle:
        parser.parse(handle.read())
    config = builder.create_builder_config()
    if fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if timing_cache_path:
        try:
            cache = config.create_timing_cache(open(timing_cache_path, "rb").read())
            config.set_timing_cache(cache, ignore_mismatch=False)
        except FileNotFoundError:
            pass
    return builder.build_serialized_network(network, config)
