// No-op signing — skip code signing during local builds
exports.default = async function (configuration) {
  // Intentionally empty — no signing required for local dev builds
};
