uniform vec4 uOrigin;
uniform vec4 uRayX;
uniform vec4 uRayY;
uniform vec4 uRayZ;
uniform vec4 uTW0;   // table_from_fulcrum row 0: R00 R01 R02 tx
uniform vec4 uTW1;   // row 1
uniform vec4 uTW2;   // row 2
out vec4 fragColor;

// per projector pixel: ray -> table plane z=0 (fulcrum frame) -> world table coords (mm)
void main()
{
    if (uOrigin.w < 0.5) {
        fragColor = TDOutputSwizzle(vec4(0.0));
        return;
    }
    vec2 pixel = vec2(vUV.x * 1280.0, (1.0 - vUV.y) * 720.0);
    vec3 ray = uRayX.xyz * pixel.x + uRayY.xyz * pixel.y + uRayZ.xyz;
    if (abs(ray.z) < 0.000001) {
        fragColor = TDOutputSwizzle(vec4(0.0));
        return;
    }
    float t = -uOrigin.z / ray.z;
    if (t <= 0.0 || t > 20000.0) {
        fragColor = TDOutputSwizzle(vec4(0.0));
        return;
    }
    vec3 p = uOrigin.xyz + t * ray;
    vec3 w = vec3(dot(uTW0.xyz, p) + uTW0.w,
                  dot(uTW1.xyz, p) + uTW1.w,
                  dot(uTW2.xyz, p) + uTW2.w);
    fragColor = TDOutputSwizzle(vec4(w.x, w.y, 0.0, 1.0));
}
