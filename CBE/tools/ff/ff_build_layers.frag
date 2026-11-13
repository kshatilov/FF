uniform vec4 uBuild;
uniform vec4 uTiming;
out vec4 fragColor;

// real DICT_4X4_50 cell patterns, face order +y(id11),+x(id12),-y(id13),-x(id14),top(id10)
const int BITS[80] = int[](
    0,0,0,1, 0,0,0,1, 1,0,1,0, 0,1,1,1,
    0,0,0,0, 1,1,1,0, 1,0,1,1, 0,1,1,1,
    0,0,1,0, 1,0,1,0, 0,0,0,0, 1,1,1,1,
    0,0,1,0, 0,1,0,0, 1,0,1,1, 0,0,0,1,
    1,1,1,1, 1,0,0,1, 1,0,0,1, 0,0,0,1);

vec3 palette(float t)
{
    return 0.5 + 0.5 * cos(6.2831853 * (t + vec3(0.0, 0.33, 0.67)));
}

void main()
{
    vec4 height = texture(sTD2DInputs[1], vUV.st);
    vec4 mark = texture(sTD2DInputs[2], vUV.st);
    if (height.a < 0.5 || mark.a < 0.5) {
        fragColor = TDOutputSwizzle(vec4(0.0));
        return;
    }
    float z = clamp(height.r, 0.0, 1.0);
    float angle = height.g;
    float radial = height.b;
    float top = smoothstep(0.97, 0.995, z);
    float breath = clamp(uBuild.x, 0.0, 1.0);
    float bands = max(2.0, uBuild.y);
    float mu = mark.r;
    float mv = mark.g;
    int face = int(floor(mark.b * 5.0));
    float mrad = length(vec2(mu, mv));
    float sdf = max(abs(mu), abs(mv)) - 1.0;

    // build timeline: layers rise bottom-up, then the finished pyramid holds
    float buildS = max(bands * 1.3, 4.0);
    float holdS = max(uTiming.x, 2.0);
    float tc = mod(uTiming.y, buildS + holdS);
    float front = clamp(tc / buildS, 0.0, 1.0);
    float fade = 1.0 - smoothstep(buildS + holdS - 0.8, buildS + holdS - 0.05, tc);
    float built = 1.0 - smoothstep(front - 0.012, front + 0.002, z);
    float drift = uTiming.y * 0.02;

    // band identity + layer seams
    float bandIdx = floor(z * bands);
    float bp = fract(z * bands);
    float seam = smoothstep(0.10, 0.02, min(bp, 1.0 - bp));
    vec3 bandCol = palette(bandIdx / bands * 0.9 + angle * 0.12 + drift);
    vec3 rgb = bandCol * (0.55 + 0.45 * breath);
    rgb += bandCol * seam * 0.55;                    // bright joint at each layer edge
    // the band currently being built flashes as it lands
    float bfp = fract(front * bands);
    float isNew = step(abs(bandIdx - floor(front * bands)), 0.5);
    rgb *= 1.0 + isNew * exp(-bfp * 5.0) * 0.9;
    // subtle marker-ripple shimmer on the built surface
    float spatial = mix(mrad * 0.45, radial * 0.55, top);
    float wave = sin(6.2831853 * (spatial * 1.5 - breath));
    rgb *= 1.0 + uBuild.z * wave * 0.35;
    rgb *= built;
    // construction front: hot scan line where the pyramid is growing
    float inBuild = step(front, 0.999);
    rgb += palette(drift + 0.4) * exp(-abs(z - front) * bands * 4.0) * 0.8 * inBuild;
    // ghost hint of the unbuilt silhouette so progress is readable
    rgb += vec3(0.012) * (0.6 + 0.4 * breath) * (1.0 - built);

    if (sdf < 0.0) {
        // inside the physical marker: 6x6 layout - black border ring stays dark,
        // white data cells (bit=1) light up only once their layer is built
        vec2 g = vec2(mu + 1.0, mv + 1.0) * 3.0;
        int gc = int(floor(g.x));
        int gr = int(floor(g.y));
        if (gc == 0 || gc == 5 || gr == 0 || gr == 5) {
            rgb = vec3(0.0);
        } else {
            int idx = (gr - 1) * 4 + (gc - 1);
            float bit = float(BITS[face * 16 + idx]);
            vec2 cf = fract(g);
            float inset = smoothstep(0.0, 0.20, cf.x) * smoothstep(1.0, 0.80, cf.x)
                        * smoothstep(0.0, 0.20, cf.y) * smoothstep(1.0, 0.80, cf.y);
            float cellPulse = 0.5 + 0.5 * sin(6.2831853 * (breath - float(idx) / 16.0));
            vec3 cellCol = palette(bandIdx / bands * 0.9 + drift + 0.13 + float(idx) * 0.03);
            rgb = mix(cellCol * 0.04, cellCol * (0.75 + 0.70 * cellPulse) * inset + cellCol * 0.03, bit);
            rgb *= built;
        }
    } else if (sdf < 0.9) {
        // the marker's own 4x4 code blooming outward as a cell frame
        vec2 g = vec2(mu + 2.0, mv + 2.0) * 1.5;
        int bc = int(clamp(floor(g.x) - 1.0, 0.0, 3.0));
        int br = int(clamp(floor(g.y) - 1.0, 0.0, 3.0));
        float bit = float(BITS[face * 16 + br * 4 + bc]);
        float cellPulse = 0.5 + 0.5 * sin(6.2831853 * (breath - length(g - 3.0) * 0.14));
        vec3 cellCol = palette(bandIdx / bands * 0.9 + drift + 0.13);
        vec3 bloom = mix(cellCol * 0.08, cellCol * (0.80 + 0.60 * cellPulse), bit);
        vec2 gf = fract(g);
        float gap = min(min(gf.x, 1.0 - gf.x), min(gf.y, 1.0 - gf.y));
        bloom *= mix(0.35, 1.0, smoothstep(0.0, 0.12, gap));
        rgb = mix(rgb, bloom, built * step(0.05, bit));
    }
    rgb = mix(rgb, vec3(1.0), smoothstep(0.16, 0.02, abs(sdf)) * step(0.0, sdf) * 0.35 * (0.5 + 0.5 * breath) * built);
    fragColor = TDOutputSwizzle(vec4(clamp(rgb * uBuild.w * fade, 0.0, 1.0), 1.0));
}
