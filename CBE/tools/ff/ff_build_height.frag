uniform vec4 uOrigin;
uniform vec4 uRayX;
uniform vec4 uRayY;
uniform vec4 uRayZ;
uniform vec4 uGeometry;
out vec4 fragColor;

bool clipPlane(vec3 normal, float bound, vec3 origin, vec3 ray, inout float nearT, inout float farT)
{
    float distance = bound - dot(normal, origin);
    float divisor = dot(normal, ray);
    if (abs(divisor) < 0.000001) return distance >= 0.0;
    float t = distance / divisor;
    if (divisor < 0.0) nearT = max(nearT, t);
    else farT = min(farT, t);
    return farT >= nearT;
}

void main()
{
    if (uOrigin.w < 0.5) {
        fragColor = TDOutputSwizzle(vec4(0.0));
        return;
    }
    vec2 pixel = vec2(vUV.x * 1280.0, (1.0 - vUV.y) * 720.0);
    vec3 origin = uOrigin.xyz;
    vec3 ray = uRayX.xyz * pixel.x + uRayY.xyz * pixel.y + uRayZ.xyz;
    float height = uGeometry.x;
    float baseHalf = uGeometry.y;
    float topHalf = uGeometry.z;
    float slope = (baseHalf - topHalf) / height;
    float nearT = 0.0;
    float farT = 10000000.0;
    bool valid = clipPlane(vec3(1, 0, slope), baseHalf, origin, ray, nearT, farT);
    valid = valid && clipPlane(vec3(-1, 0, slope), baseHalf, origin, ray, nearT, farT);
    valid = valid && clipPlane(vec3(0, 1, slope), baseHalf, origin, ray, nearT, farT);
    valid = valid && clipPlane(vec3(0, -1, slope), baseHalf, origin, ray, nearT, farT);
    valid = valid && clipPlane(vec3(0, 0, 1), height, origin, ray, nearT, farT);
    valid = valid && clipPlane(vec3(0, 0, -1), 0.0, origin, ray, nearT, farT);
    vec3 hit = origin + nearT * ray;
    float z = clamp(hit.z / height, 0.0, 1.0);
    float angle = fract(atan(hit.y, hit.x) / 6.2831853 + 0.125);
    float halfEdge = max(mix(baseHalf, topHalf, z), 0.001);
    float radial = clamp(length(hit.xy) / halfEdge, 0.0, 1.0);
    fragColor = TDOutputSwizzle(valid ? vec4(z, angle, radial, 1.0) : vec4(0.0));
}
