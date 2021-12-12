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



constexpr int N = 400;
constexpr int M = 1995;

using namespace std;

template< typename T = int >
struct Edge {
    int from, to;
    T cost;
    int idx;
    int status; // 0: not fixed, 1: fixed(use), -1: fixed(not use)

    Edge() = default;

    Edge(int from, int to, T cost = 1, int idx = -1, int status = 0) : from(from), to(to), cost(cost), idx(idx), status(status) {}

    string stringify() const {
        std::ostringstream oss;
        oss << "Edge [from=" << from << ", to=" << to << ", cost=" << cost << ", idx=" << idx << ", status=%d" << status << "]";
        return oss.str();
    }

    operator int() const { return to; }
};

template< typename T = int >
struct Graph {
    vector< vector< Edge< T > > > g;
    int es;

    Graph() = default;

    explicit Graph(int n) : g(n), es(0) {}

    size_t size() const {
        return g.size();
    }

    void add_directed_edge(int from, int to, T cost = 1) {
        g[from].emplace_back(from, to, cost, es++);
    }

    void add_edge(int from, int to, T cost = 1) {
        g[from].emplace_back(from, to, cost, es);
        g[to].emplace_back(to, from, cost, es++);
    }

    void read(int M, int padding = -1, bool weighted = false, bool directed = false) {
        for (int i = 0; i < M; i++) {
            int a, b;
            cin >> a >> b;
            a += padding;
            b += padding;
            T c = T(1);
            if (weighted) cin >> c;
            if (directed) add_directed_edge(a, b, c);
            else add_edge(a, b, c);
        }
    }

    inline vector< Edge< T > >& operator[](const int& k) {
        return g[k];
    }

    inline const vector< Edge< T > >& operator[](const int& k) const {
        return g[k];
    }
};

template< typename T = int >
using Edges = vector< Edge< T > >;

struct UnionFind {
    vector< int > data;

    UnionFind() = default;

    explicit UnionFind(size_t sz) : data(sz, -1) {}

    bool unite(int x, int y) {
        x = find(x), y = find(y);
        if (x == y) return false;
        if (data[x] > data[y]) swap(x, y);
        data[x] += data[y];
        data[y] = x;
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

template< typename T >
struct MinimumSpanningTree {
    T cost;
    Edges< T > edges;
};

template< typename T >
MinimumSpanningTree< T > kruskal(Edges< T >& edges, int V) {
    sort(begin(edges), end(edges), [](const Edge< T >& a, const Edge< T >& b) {
        return a.cost < b.cost;
        });
    UnionFind tree(V);
    T total = T();
    Edges< T > es;
    for (auto& e : edges) {
        if (tree.unite(e.from, e.to)) {
            es.emplace_back(e);
            total += e.cost;
        }
    }
    return { total, es };
}

template< typename T >
MinimumSpanningTree< T > kruskal2(Edges<T> edges, int V) {

    T total = T();
    UnionFind tree(V);

    Edges<T> es;
    Edges<T> free_edges;

    for (const auto& e : edges) {
        if (e.status == 0) {
            free_edges.push_back(e);
        }
        else if (e.status == 1) {
            // use
            tree.unite(e.from, e.to);
            es.push_back(e);
            total += e.cost;
        }
    }

    sort(begin(free_edges), end(free_edges), [](const Edge< T >& a, const Edge< T >& b) {
        return a.cost < b.cost;
        });

    for (auto& e : free_edges) {
        if (tree.unite(e.from, e.to)) {
            es.emplace_back(e);
            total += e.cost;
        }
    }
    return { total, es };

}

int solve(istream& in, ostream& out) {
    using pii = pair<int, int>;

    vector<pii> points;
    Edges<double> edges;
    vector<double> base_distances(M);

    auto calc_coeff = [&](int eid, double start_coeff, double end_coeff) {
        double progress = (double)eid / (M - 1);
        return start_coeff * (1.0 - progress) + end_coeff * progress;
    };

    points.resize(N);
    for (int i = 0; i < N; i++) {
        in >> points[i].first >> points[i].second;
    }
    for (int i = 0; i < M; i++) {
        int u, v;
        in >> u >> v;
        auto [ux, uy] = points[u];
        auto [vx, vy] = points[v];

        base_distances[i] = (int)round(sqrt(pow(abs(ux - vx), 2.0) + pow(abs(uy - vy), 2.0)));
        edges.emplace_back(u, v, base_distances[i] * calc_coeff(i, 1.72, 1.79), i); // [d, 3d] ‚ÌŠú‘Ò’l‚Í 2d
    }

    double actual_cost;

    for (int i = 0; i < M; i++) {
        int len;
        in >> len;
        edges[i].cost = len; // true value

        auto [cost, es] = kruskal2(edges, N);
        bitset<M> to_use;
        for (const auto& e : es) {
            to_use[e.idx] = true;
        }

        if (to_use[i]) {
            edges[i].status = 1;
            out << 1 << endl;
        }
        else {
            edges[i].status = -1;
            out << 0 << endl;
        }

        if (i == M - 1) actual_cost = cost;
    }

    auto [theoretical_cost, _] = kruskal(edges, N);

    return (int)round(1e8 * theoretical_cost / actual_cost);
}

#ifdef _MSC_VER
long long batch_test() {

    vector<long long> scores(150, 0);

    concurrency::parallel_for(0, 150, [&](int seed) {
        string in_file = format("C:\\dev\\heuristic\\tasks\\AHC007\\tools\\in\\%04d.txt", seed);
        string out_file = format("C:\\dev\\heuristic\\tasks\\AHC007\\tools\\out\\%04d.txt", seed);
        ifstream ifs(in_file);
        ofstream ofs(out_file);
        int score = solve(ifs, ofs);
        scores[seed] = score;
    });

    long long score_sum = accumulate(scores.begin(), scores.end(), 0LL);
    dump(score_sum);
    return score_sum;
}

int main() {
    batch_test();
}

#else

int main() {

    std::istream& in = std::cin;
    std::ostream& out = std::cout;

    solve(in, out);

    return 0;
}

#endif