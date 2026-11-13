uniform vec3 uTableGain;
uniform vec3 uTableOptions;
out vec4 fragColor;

// Camera-relative table white balance, applied after the geometric warp.
// Gains include measured headroom; this adds no light and never lifts black.
void main()
{
    vec3 original = clamp(texture(sTD2DInputs[0], vUV.st).rgb, 0.0, 1.0);
    vec3 corrected = original * clamp(uTableGain, 0.0, 1.0);
    float strength = clamp(uTableOptions.x, 0.0, 1.0);
    fragColor = TDOutputSwizzle(
        vec4(mix(original, corrected, strength), 1.0));
}
