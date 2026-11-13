uniform vec4 uBuild;
uniform vec4 uTiming;
uniform vec4 uTW0;   // table_from_fulcrum row 0: R00 R01 R02 tx
uniform vec4 uTW1;   // row 1
uniform vec4 uTW2;   // row 2
out vec4 fragColor;

// marker arrival: polyomino pieces from the real DICT_4X4_50 patterns, cell -> piece (-1 empty)
const int PIECES[80] = int[](
    -1,-1,-1,2, -1,-1,-1,2, 1,-1,0,-1, -1,0,0,0,
    -1,-1,-1,-1, 0,0,0,-1, 0,-1,0,0, -1,0,0,0,
    -1,-1,1,-1, 2,-1,1,-1, -1,-1,-1,-1, 0,0,0,0,
    -1,-1,3,-1, -1,2,-1,-1, 1,-1,0,0, -1,-1,-1,0,
    0,0,0,0, 0,-1,-1,0, 0,-1,-1,0, -1,-1,-1,0);
const int NP[5] = int[](3, 1, 3, 4, 1);
// white cells in flight order (cell -> order, -1 empty)
const int ORDER[80] = int[](
    -1,-1,-1,0, -1,-1,-1,1, 2,-1,3,-1, -1,4,5,6,
    -1,-1,-1,-1, 0,1,2,-1, 3,-1,4,5, -1,6,7,8,
    -1,-1,0,-1, 1,-1,2,-1, -1,-1,-1,-1, 3,4,5,6,
    -1,-1,0,-1, -1,1,-1,-1, 2,-1,3,4, -1,-1,-1,5,
    0,1,2,3, 4,-1,-1,5, 6,-1,-1,7, -1,-1,-1,8);
const int WBS[50] = int[](
    3,7,8,10,13,14,15,-1,-1,-1,
    4,5,6,8,10,11,13,14,15,-1,
    2,4,6,12,13,14,15,-1,-1,-1,
    2,5,8,10,11,15,-1,-1,-1,-1,
    0,1,2,3,4,7,8,11,15,-1);
const int WCOUNT[5] = int[](7, 9, 7, 6, 9);
const int WBASE[5] = int[](0, 7, 16, 23, 29);
const int WSUM = 38;
// ArUco marker frames from fulcrum_spec.json (spec),
// face order +y(id11),+x(id12),-y(id13),-x(id14),top(id10); MH = (half_u, half_v)
const vec3 MC[5] = vec3[](vec3(0.0,57.375,21.385), vec3(57.375,0.0,21.385),
                          vec3(0.0,-57.375,21.385), vec3(-57.375,0.0,21.385), vec3(0.0,0.0,42.5));
const vec3 MU[5] = vec3[](vec3(-1.0,0.0,0.0), vec3(0.0,1.0,0.0),
                          vec3(1.0,0.0,0.0), vec3(0.0,-1.0,0.0), vec3(1.0,0.0,0.0));
const vec3 MV[5] = vec3[](vec3(0.0,0.68538,-0.72818), vec3(0.68538,0.0,-0.72818),
                          vec3(0.0,-0.68538,-0.72818), vec3(-0.68538,0.0,-0.72818), vec3(0.0,-1.0,0.0));
const vec2 MH[5] = vec2[](vec2(25.0,25.0), vec2(25.0,25.0), vec2(25.0,25.0),
                          vec2(25.0,25.0), vec2(27.5,27.5));
// table well: TCOLS x TROWS cells just south (-y world) of the ff's current table position;
// the well follows the ff, so where the ff sits decides where blocks land
const int TCOLS = 18;
const int TROWS = 6;
const float TCELL = 10.0;
const float TMARGIN = 88.0;
const int SLOT_CAP = 108;   // TCOLS * TROWS
const int PILE_GR = 3;      // pile cycles per fill (rows clear as they complete)

vec3 worldOf(vec3 p)
{
    return vec3(dot(uTW0.xyz, p) + uTW0.w,
                dot(uTW1.xyz, p) + uTW1.w,
                dot(uTW2.xyz, p) + uTW2.w);
}

// cell order within a row: center column first, then alternating outward;
// the j range is asymmetric (-TCOLS/2 .. TCOLS/2-1) so the far-left cell orders last
int rowOrder(int j)
{
    if (j == 0) return 0;
    if (j > 0) return 2 * j - 1;
    int d = -j;
    return d == TCOLS / 2 ? TCOLS - 1 : 2 * d;
}
int orderCol(int s)
{
    if (s == 0) return 0;
    if (s == TCOLS - 1) return -TCOLS / 2;
    return ((s & 1) == 1) ? (s + 1) / 2 : -s / 2;
}

float cellEdge(vec2 f)
{
    float th = 0.10;
    return max(max(smoothstep(th, 0.0, f.x), smoothstep(1.0 - th, 1.0, f.x)),
               max(smoothstep(th, 0.0, f.y), smoothstep(1.0 - th, 1.0, f.y)));
}

void main()
{
    vec4 mark = texture(sTD2DInputs[1], vUV.st);
    vec4 tbl = texture(sTD2DInputs[2], vUV.st);
    float CYC = max(uTiming.x, 7.5);
    float stepT = 0.35, fall = 0.55, M0 = 2.4, moveStep = 0.32, moveDur = 0.6;
    int G = int(floor(uTiming.y / CYC));
    int pileCyc = int(mod(float(G), float(PILE_GR)));
    int pileBase = pileCyc * WSUM;      // first shared rank of this pile cycle
    float pileT = mod(uTiming.y, CYC);
    float fx = uTW0.w, fy = uTW1.w;
    vec3 col = vec3(0.0);

    // per-face departure state in this cycle (each face runs on its own 0.35s offset)
    int slots[5];
    int landed[5];
    for (int f = 0; f < 5; ++f) {
        float tf = mod(uTiming.y + float(f) * 0.35, CYC);
        slots[f] = min(WCOUNT[f], max(0, SLOT_CAP - pileBase - WBASE[f]));
        landed[f] = int(clamp(floor((tf - M0 - moveDur) / moveStep) + 1.0, 0.0, float(slots[f])));
    }

    // ============ ff surface: pieces drop into the ArUco codes ============
    if (mark.a > 0.5) {
        float mu = mark.r;
        float mv = mark.g;
        int face = int(floor(mark.b * 5.0));
        int cyc = int(mod(float(G), 8.0));
        float t = mod(uTiming.y + float(face) * 0.35, CYC);
        float fadeAll = (cyc == 7) ? 1.0 - smoothstep(CYC - 0.8, CYC - 0.05, t) : 1.0;
        int slotsFace = slots[face];
        vec2 g = vec2(mu + 1.0, mv + 1.0) * 3.0;
        if (g.x >= 1.0 && g.x < 5.0 && g.y >= 1.0 && g.y < 5.0) {
            vec2 p = g - 1.0;
            for (int pi = 0; pi < NP[face]; ++pi) {
                float pr = clamp((t - float(pi) * stepT) / fall, 0.0, 1.0);
                float s = (1.0 - pow(pr, 1.7)) * 4.5;
                for (int pass = 0; pass < 2; ++pass) {
                    vec2 q = vec2(p.x, p.y + s + (pass == 1 && s > 0.02 ? 0.5 : 0.0));
                    int c = int(floor(q.x)), r = int(floor(q.y));
                    if (r < 0 || r > 3 || c < 0 || c > 3) continue;
                    int idx = face * 16 + r * 4 + c;
                    if (PIECES[idx] != pi) continue;
                    int dep_i = ORDER[idx];
                    if (dep_i >= 0 && dep_i < slotsFace && t > M0 + float(dep_i) * moveStep) continue;
                    vec2 f = fract(q);
                    float th = 0.12, e = 0.0;
                    if (f.x < th && (c - 1 < 0 || PIECES[idx - 1] != pi ||
                        (ORDER[idx - 1] >= 0 && ORDER[idx - 1] < slotsFace && t > M0 + float(ORDER[idx - 1]) * moveStep))) e = 1.0;
                    if (f.x > 1.0 - th && (c + 1 > 3 || PIECES[idx + 1] != pi ||
                        (ORDER[idx + 1] >= 0 && ORDER[idx + 1] < slotsFace && t > M0 + float(ORDER[idx + 1]) * moveStep))) e = 1.0;
                    if (f.y < th && (r - 1 < 0 || PIECES[idx - 4] != pi ||
                        (ORDER[idx - 4] >= 0 && ORDER[idx - 4] < slotsFace && t > M0 + float(ORDER[idx - 4]) * moveStep))) e = 1.0;
                    if (f.y > 1.0 - th && (r + 1 > 3 || PIECES[idx + 4] != pi ||
                        (ORDER[idx + 4] >= 0 && ORDER[idx + 4] < slotsFace && t > M0 + float(ORDER[idx + 4]) * moveStep))) e = 1.0;
                    float flash = exp(-max(t - float(pi) * stepT - fall, 0.0) * 6.0) * step(fall, t - float(pi) * stepT);
                    col += vec3(pass == 0 ? 0.06 + e * (0.85 + 1.8 * flash) : 0.02 + e * 0.30) * fadeAll;
                }
            }
        }
    }

    // ============ table surface: shared well south of the ff ============
    float fadePile = (pileCyc == PILE_GR - 1) ? 1.0 - smoothstep(CYC - 0.8, CYC - 0.05, pileT) : 1.0;
    if (tbl.a > 0.5) {
        float wx = tbl.r;
        float wy = tbl.g;
        float sy = fy - TMARGIN - wy;      // depth into the build zone
        if (sy > 0.0) {
            int rr = int(floor(sy / TCELL));
            if (rr < TROWS) {
                int j = int(floor((wx - fx) / TCELL + 0.5));
                if (j >= -TCOLS / 2 && j < TCOLS / 2) {
                    int rank = rr * TCOLS + rowOrder(j);
                    int rel = rank - pileBase;
                    int owner = -1;
                    for (int f = 0; f < 5; ++f)
                        if (rel >= WBASE[f] && rel < WBASE[f] + WCOUNT[f]) owner = f;
                    bool occupied = owner >= 0 && rel - WBASE[owner] < landed[owner];
                    bool rowFull = true;
                    float rowTc = -100.0;
                    for (int k = 0; k < TCOLS; ++k) {
                        int rk = rr * TCOLS + k;
                        int relk = rk - pileBase;
                        int own = -1;
                        for (int f = 0; f < 5; ++f)
                            if (relk >= WBASE[f] && relk < WBASE[f] + WCOUNT[f]) own = f;
                        if (own < 0 || relk - WBASE[own] >= landed[own]) { rowFull = false; break; }
                        int ik = relk - WBASE[own];
                        float depT = M0 + float(ik) * moveStep + moveDur - float(own) * 0.35;
                        rowTc = max(rowTc, depT);
                    }
                    vec2 fr = vec2(fract((wx - fx) / TCELL + 0.5), fract(sy / TCELL));
                    float ed = cellEdge(fr);
                    if (occupied && !rowFull) {
                        col += vec3(0.04 + ed * 0.85) * fadePile;
                    } else if (rowFull && pileT >= rowTc && pileT < rowTc + 0.7) {
                        float flash = exp(-(pileT - rowTc) * 5.0);
                        col += vec3(0.10 + ed * 1.4 + 0.6 * flash) * flash;
                    }
                }
            }
        }

        // ============ blocks in transit: marker cell -> table slot ============
        for (int f = 0; f < 5; ++f) {
            float tf = mod(uTiming.y + float(f) * 0.35, CYC);
            for (int i = 0; i < slots[f]; ++i) {
                float dep = M0 + float(i) * moveStep;
                if (tf < dep || tf > dep + moveDur) continue;
                float e = smoothstep(0.0, 1.0, (tf - dep) / moveDur);
                int cell = WBS[f * 10 + i];
                int c = cell % 4, r = cell / 4;
                vec3 lp = MC[f] + ((float(c) - 1.5) / 3.0 * MH[f].x) * MU[f]
                                + ((float(r) - 1.5) / 3.0 * MH[f].y) * MV[f];
                vec3 sw = worldOf(lp);
                int rank = pileBase + WBASE[f] + i;
                int rr2 = rank / TCOLS;
                int jj = orderCol(int(mod(float(rank), float(TCOLS))));
                vec2 dst = vec2(fx + float(jj) * TCELL,
                                fy - TMARGIN - float(rr2) * TCELL - TCELL * 0.5);
                for (int pass = 0; pass < 2; ++pass) {
                    vec2 pos = mix(sw.xy, dst, pass == 1 ? max(e - 0.18, 0.0) : e);
                    vec2 d = abs(vec2(wx, wy) - pos);
                    if (d.x < TCELL * 0.5 && d.y < TCELL * 0.5) {
                        vec2 fr = (vec2(wx, wy) - pos) / TCELL + 0.5;
                        float ed = cellEdge(fr);
                        float amp = pass == 0 ? 1.0 : 0.3;
                        col += vec3((0.05 + ed * 0.9) * amp);
                    }
                }
            }
        }
    }

    fragColor = TDOutputSwizzle(vec4(clamp(col * uBuild.w * (0.7 + 0.3 * uBuild.x), 0.0, 1.0), 1.0));
}
