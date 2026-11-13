uniform vec3 uH0;
uniform vec3 uH1;
uniform vec3 uH2;
out vec4 fragColor;

// Pull sampling: each projector UV asks which raw camera UV it corresponds to.
// Both UV spaces have a bottom-left origin; hg_geometry handles pixel centres.
void main()
{
    vec3 p = vec3(vUV.st, 1.0);
    vec3 q = vec3(dot(uH0, p), dot(uH1, p), dot(uH2, p));
    vec4 color = vec4(0.0, 0.0, 0.0, 1.0);
    if (abs(q.z) > 1e-10) {
        vec2 uv = q.xy / q.z;
        if (all(greaterThanEqual(uv, vec2(0.0))) &&
            all(lessThanEqual(uv, vec2(1.0)))) {
            color = vec4(texture(sTD2DInputs[0], uv).rgb, 1.0);
        }
    }
    fragColor = TDOutputSwizzle(color);
}
