uniform vec4 uOrigin;
uniform vec4 uRayX;
uniform vec4 uRayY;
uniform vec4 uRayZ;
uniform vec4 uGeometry;
out vec4 fragColor;

// ArUco marker frames from fulcrum_spec.json,
// face order +y(id11),+x(id12),-y(id13),-x(id14),top(id10); MH = (half_u, half_v)
const vec3 MC[5] = vec3[](vec3(0.0,57.375,21.385), vec3(57.375,0.0,21.385),
                          vec3(0.0,-57.375,21.385), vec3(-57.375,0.0,21.385), vec3(0.0,0.0,42.5));
const vec3 MU[5] = vec3[](vec3(-1.0,0.0,0.0), vec3(0.0,1.0,0.0),
                          vec3(1.0,0.0,0.0), vec3(0.0,-1.0,0.0), vec3(1.0,0.0,0.0));
const vec3 MV[5] = vec3[](vec3(0.0,0.68538,-0.72818), vec3(0.68538,0.0,-0.72818),
                          vec3(0.0,-0.68538,-0.72818), vec3(-0.68538,0.0,-0.72818), vec3(0.0,-1.0,0.0));
const vec2 MH[5] = vec2[](vec2(25.0,25.0), vec2(25.0,25.0), vec2(25.0,25.0),
                          vec2(25.0,25.0), vec2(27.5,27.5));

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
    int face = 4;
    if (valid && z < 0.985) {
        float e = max(mix(baseHalf, topHalf, z), 0.001);
        face = abs(abs(hit.y) - e) < abs(abs(hit.x) - e) ? (hit.y > 0.0 ? 0 : 2) : (hit.x > 0.0 ? 1 : 3);
    }
    vec3 rel = hit - MC[face];
    float u = dot(rel, MU[face]) / MH[face].x;
    float v = dot(rel, MV[face]) / MH[face].y;
    fragColor = TDOutputSwizzle(valid ? vec4(u, v, (float(face) + 0.5) / 5.0, 1.0) : vec4(0.0));
}
