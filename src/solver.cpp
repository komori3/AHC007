#include <bits/stdc++.h>
#include <random>
#ifdef _MSC_VER
#include <ppl.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#else
#pragma GCC target("avx2")
#pragma GCC optimize("O3")
#pragma GCC optimize("unroll-loops")
#endif

/** compro_io **/

/* tuple */
// out
namespace aux {
    template<typename T, unsigned N, unsigned L>
    struct tp {
        static void output(std::ostream& os, const T& v) {
            os << std::get<N>(v) << ", ";
            tp<T, N + 1, L>::output(os, v);
        }
    };
    template<typename T, unsigned N>
    struct tp<T, N, N> {
        static void output(std::ostream& os, const T& v) { os << std::get<N>(v); }
    };
}
template<typename... Ts>
std::ostream& operator<<(std::ostream& os, const std::tuple<Ts...>& t) {
    os << '[';
    aux::tp<std::tuple<Ts...>, 0, sizeof...(Ts) - 1>::output(os, t);
    return os << ']';
}

template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x);

/* pair */
// out
template<class S, class T>
std::ostream& operator<<(std::ostream& os, const std::pair<S, T>& p) {
    return os << "[" << p.first << ", " << p.second << "]";
}
// in
template<class S, class T>
std::istream& operator>>(std::istream& is, const std::pair<S, T>& p) {
    return is >> p.first >> p.second;
}

/* container */
// out
template<class Ch, class Tr, class Container>
std::basic_ostream<Ch, Tr>& operator<<(std::basic_ostream<Ch, Tr>& os, const Container& x) {
    bool f = true;
    os << "[";
    for (auto& y : x) {
        os << (f ? "" : ", ") << y;
        f = false;
    }
    return os << "]";
}
// in
template <
    class T,
    class = decltype(std::begin(std::declval<T&>())),
    class = typename std::enable_if<!std::is_same<T, std::string>::value>::type
>
std::istream& operator>>(std::istream& is, T& a) {
    for (auto& x : a) is >> x;
    return is;
}

/* struct */
template<typename T>
auto operator<<(std::ostream& out, const T& t) -> decltype(out << t.stringify()) {
    out << t.stringify();
    return out;
}

/* setup */
struct IOSetup {
    IOSetup(bool f) {
        if (f) { std::cin.tie(nullptr); std::ios::sync_with_stdio(false); }
        std::cout << std::fixed << std::setprecision(15);
    }
} iosetup(true);

/** string formatter **/
template<typename... Ts>
std::string format(const std::string& f, Ts... t) {
    size_t l = std::snprintf(nullptr, 0, f.c_str(), t...);
    std::vector<char> b(l + 1);
    std::snprintf(&b[0], l + 1, f.c_str(), t...);
    return std::string(&b[0], &b[0] + l);
}

template<typename T>
std::string stringify(const T& x) {
    std::ostringstream oss;
    oss << x;
    return oss.str();
}

/* dump */
#define ENABLE_DUMP
#ifdef ENABLE_DUMP
#define DUMPOUT std::cerr
std::ostringstream DUMPBUF;
#define dump(...) do{DUMPBUF<<"  ";DUMPBUF<<#__VA_ARGS__<<" :[DUMP - "<<__LINE__<<":"<<__FUNCTION__<<"]"<<std::endl;DUMPBUF<<"    ";dump_func(__VA_ARGS__);DUMPOUT<<DUMPBUF.str();DUMPBUF.str("");DUMPBUF.clear();}while(0);
void dump_func() { DUMPBUF << std::endl; }
template <class Head, class... Tail> void dump_func(Head&& head, Tail&&... tail) { DUMPBUF << head; if (sizeof...(Tail) == 0) { DUMPBUF << " "; } else { DUMPBUF << ", "; } dump_func(std::move(tail)...); }
#else
#define dump(...) void(0);
#endif

/* timer */
class Timer {
    double t = 0, paused = 0, tmp;
public:
    Timer() { reset(); }
    static double time() {
#ifdef _MSC_VER
        return __rdtsc() / 3.0e9;
#else
        unsigned long long a, d;
        __asm__ volatile("rdtsc"
            : "=a"(a), "=d"(d));
        return (d << 32 | a) / 3.0e9;
#endif
    }
    void reset() { t = time(); }
    void pause() { tmp = time(); }
    void restart() { paused += time() - tmp; }
    double elapsed_ms() { return (time() - t - paused) * 1000.0; }
} timer;

/* rand */
struct Xorshift {
    uint64_t x = 88172645463325252LL;
    void set_seed(unsigned seed, int rep = 100) { x = uint64_t((seed + 1) * 10007); for (int i = 0; i < rep; i++) next_int(); }
    unsigned next_int() { x = x ^ (x << 7); return x = x ^ (x >> 9); }
    unsigned next_int(unsigned mod) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % mod; }
    unsigned next_int(unsigned l, unsigned r) { x = x ^ (x << 7); x = x ^ (x >> 9); return x % (r - l + 1) + l; } // inclusive
    double next_double() { return double(next_int()) / UINT_MAX; }
} rnd;

/* shuffle */
template<typename T>
void shuffle_vector(std::vector<T>& v, Xorshift& rnd) {
    int n = v.size();
    for (int i = n - 1; i >= 1; i--) {
        int r = rnd.next_int(i);
        std::swap(v[i], v[r]);
    }
}

/* split */
std::vector<std::string> split(std::string str, const std::string& delim) {
    for (char& c : str) if (delim.find(c) != std::string::npos) c = ' ';
    std::istringstream iss(str);
    std::vector<std::string> parsed;
    std::string buf;
    while (iss >> buf) parsed.push_back(buf);
    return parsed;
}

template<typename A, size_t N, typename T> inline void Fill(A(&array)[N], const T& val) {
    std::fill((T*)array, (T*)(array + N), val);
}

template<typename T> bool chmax(T& a, const T& b) { if (a < b) { a = b; return true; } return false; }
template<typename T> bool chmin(T& a, const T& b) { if (a > b) { a = b; return true; } return false; }



using namespace std;

constexpr int N = 400;
constexpr int M = 1995;

struct TestCase {
    int x[N], y[N];
    int u[M], v[M];
    TestCase() = default;
    TestCase(istream& in) {
        for (int i = 0; i < N; i++) {
            in >> x[i] >> y[i];
        }
        for (int i = 0; i < M; i++) {
            in >> u[i] >> v[i];
        }
    }
};

struct UnionFind {

    int num_sets;
    vector< int > data;

    UnionFind() = default;

    explicit UnionFind(size_t sz) : data(sz, -1), num_sets(sz) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) swap(x, y);
        data[x] += data[y];
        data[y] = x;
        num_sets--;
        return true;
    }

    int find(int k) {
        if (data[k] < 0) return (k);
        return data[k] = find(data[k]);
    }

    int size(int k) {
        return -data[find(k)];
    }

    bool same(int x, int y) {
        return find(x) == find(y);
    }

    vector< vector< int > > groups() {
        int n = (int)data.size();
        vector< vector< int > > ret(n);
        for (int i = 0; i < n; i++) {
            ret[find(i)].emplace_back(i);
        }
        ret.erase(remove_if(begin(ret), end(ret), [&](const vector< int >& v) {
            return v.empty();
            }));
        return ret;
    }
};

struct State {
    Xorshift rnd;
    int turn;
    int xs[N], ys[N];
    int us[M], vs[M];
    int base_dist[M];
    int actual_dist[M];
    bitset<M> used;
    UnionFind tree;
    int cost;
    State(const TestCase& tc) : tree(N) {
        turn = 0;
        memcpy(xs, tc.x, sizeof(int) * N);
        memcpy(ys, tc.y, sizeof(int) * N);
        memcpy(us, tc.u, sizeof(int) * M);
        memcpy(vs, tc.v, sizeof(int) * M);
        for (int i = 0; i < M; i++) {
            int u = tc.u[i], v = tc.v[i];
            int ux = tc.x[u], uy = tc.y[u];
            int vx = tc.x[v], vy = tc.y[v];
            base_dist[i] = (int)round(sqrt((ux - vx) * (ux - vx) + (uy - vy) * (uy - vy)));
            used[i] = 0;
        }
        cost = 0;
    }
    int query(int len) {
        using pii = pair<int, int>;

        actual_dist[turn] = len;
        int u = us[turn], v = vs[turn];
        if (tree.same(u, v)) return used[turn++] = false;

        // 未決定辺の長さをランダムに設定して mst を生成することを繰り返す

        // 辺 e が最小全域木で使われる
        // -> 辺 e よりコストの小さい辺を（閉路ができるかどうかに関わらず）UnionFind で unite した際に、e の両端点が連結していない

        int num_trial = 100;
        int num_selected = 0;
        for (int t = 0; t < num_trial; t++) {
            bool use_flag = true;
            UnionFind ctree(tree);
            // 未決定辺にランダムに重み付け
            for (int j = turn + 1; j < M; j++) {
                int d = rnd.next_int(base_dist[j], base_dist[j] * 3);
                if (d >= len) continue;
                ctree.unite(us[j], vs[j]);
                if (ctree.same(u, v)) {
                    use_flag = false;
                    break;
                }
            }
            num_selected += use_flag;
        }

        used[turn] = num_selected >= num_trial * 4 / 10;
        if (used[turn]) {
            tree.unite(u, v);
            cost += len;
        }

        return used[turn++];
    }

    int calc_true_mst_cost() const {
        using pii = pair<int, int>;
        UnionFind ctree(N);
        vector<pii> edges;
        for (int i = 0; i < M; i++) {
            edges.emplace_back(actual_dist[i], i);
        }
        sort(edges.begin(), edges.end());
        int mst_cost = 0;
        for (const auto [d, i] : edges) {
            if (ctree.unite(us[i], vs[i])) {
                mst_cost += d;
            }
        }
        return mst_cost;
    }

    int calc_score() const {
        return (int)round(1e8 * calc_true_mst_cost() / cost);
    }
};

int solve(istream& in, ostream& out, bool no_output = false) {
    Timer timer;
    using pii = pair<int, int>;

    const TestCase tc(in);

    State state(tc);

    for (int i = 0; i < M; i++) {
        int len;
        in >> len;
        int res = state.query(len);
        if (!no_output) out << res << endl;
    }
    dump(timer.elapsed_ms(), state.calc_score());
    return state.calc_score();
}

#ifdef _MSC_VER
long long batch_test(int num_seeds) {

    vector<long long> scores(num_seeds, 0);

    concurrency::parallel_for(0, num_seeds, [&](int seed) {
        string in_file = format("C:\\dev\\heuristic\\tasks\\AHC007\\tools\\in\\%04d.txt", seed);
        ifstream ifs(in_file);
        int score = solve(ifs, cout, true);
        scores[seed] = score;
        }, concurrency::simple_partitioner(num_seeds / 10));

    long long score_sum = accumulate(scores.begin(), scores.end(), 0LL);
    dump(score_sum);
    return score_sum;
}

int single_test() {
    string in_file = "C:\\dev\\heuristic\\tasks\\AHC007\\tools\\in\\0000.txt";
    string out_file = "C:\\dev\\heuristic\\tasks\\AHC007\\tools\\out\\0000.txt";
    ifstream ifs(in_file);
    ofstream ofs(out_file);
    int score = solve(ifs, ofs);
    dump(score);
}

int main() {
    batch_test(150);
    //single_test();
}

#else

int main() {

    std::istream& in = std::cin;
    std::ostream& out = std::cout;

    solve(in, out);

    return 0;
}

#endif