// Baseline notes for turning fused_relu_plugin.hip into a real MIGraphX plugin.
//
// Implement and record:
// - Plugin creator fields: alpha, version, namespace, field collection.
// - Shape inference: output dims must exactly match input dims.
// - Format support: FP32/FP16 and linear formats only until tested otherwise.
// - Serialization: every runtime parameter must round-trip exactly.
// - Workspace: return zero unless enqueue uses temporary storage.
// - Enqueue: use MIGraphX-provided hipStream_t and return nonzero on launch
//   failure, without inserting device-wide synchronization.
// - Engine test: build a one-plugin network and compare to framework output.

int migraphx_plugin_lifecycle_notes_anchor() {
  return 0;
}
