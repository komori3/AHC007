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



constexpr int N = 1000;
constexpr int M = 50;

//using Point = std::pair<int, int>;

struct Point {

    int x, y;

    Point(int x = 0, int y = 0) : x(x), y(y) {}

    inline int distance(const Point& p) {
        return abs(x - p.x) + abs(y - p.y);
    }

};

struct Order {
    int id;
    Point from, to;
};

struct TestCase {

    std::array<int, N> a;
    std::array<int, N> b;
    std::array<int, N> c;
    std::array<int, N> d;

    TestCase(std::istream& in) {
        for (int i = 0; i < N; i++) {
            in >> a[i] >> b[i] >> c[i] >> d[i];
        }
    }

    std::vector<Order> get_orders() const {
        std::vector<Order> orders;
        for (int id = 0; id < N; id++) {
            Order order;
            order.id = id;
            order.from.x = a[id];
            order.from.y = b[id];
            order.to.x = c[id];
            order.to.y = d[id];
            orders.push_back(order);
        }
        return orders;
    }

};

struct Solution {
    int total_dist;
    std::vector<int> order_ids;
    std::vector<Point> route;

    Solution(int total_dist, const std::vector<int>& order_ids, const std::vector<Point>& route) : total_dist(total_dist), order_ids(order_ids), route(route) {}

    std::string stringify() const {
        std::ostringstream oss;
        oss << order_ids.size();
        for (int r : order_ids) {
            oss << ' ' << r + 1;
        }
        oss << '\n';
        oss << route.size();
        for (const auto& [x, y] : route) {
            oss << ' ' << x << ' ' << y;
        }
        return oss.str();
    }
};

struct Solver {

    TestCase tc;

    Solver(const TestCase& tc) : tc(tc) {}

    Solution solve_sub(const std::vector<Order>& sub_orders) {

        Point origin(400, 400);

        // nearest neighbor
        std::bitset<M> picked; // picked ‚È‚ç to ‚àŒó•â‚É“ü‚é
        std::bitset<M> delivered;

        int total_dist = 0;
        std::vector<Point> route({ origin });
        Point pos(origin);
        while (true) {

            int min_dist = INT_MAX;
            int min_idx = -1;
            bool is_pick = false;
            for (int i = 0; i < M; i++) {
                const auto& order = sub_orders[i];
                if (!picked[i]) {
                    // pick
                    int dist = pos.distance(order.from);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = i;
                        is_pick = true;
                    }
                }
                else if (!delivered[i]) {
                    // deliver
                    int dist = pos.distance(order.to);
                    if (dist < min_dist) {
                        min_dist = dist;
                        min_idx = i;
                        is_pick = false;
                    }
                }
            }

            (is_pick ? picked : delivered)[min_idx] = true;
            auto npos = is_pick ? sub_orders[min_idx].from : sub_orders[min_idx].to;
            route.push_back(npos);
            total_dist += pos.distance(npos);
            pos = npos;

            if (delivered.count() == M) break;
        }
        route.push_back(origin);
        total_dist += origin.distance(pos);

        std::vector<int> order_ids;
        for (const auto& order : sub_orders) {
            order_ids.push_back(order.id);
        }

        return Solution(total_dist, order_ids, route);
    }

    Solution solve() {

        Point origin(400, 400);
        
        auto orders = tc.get_orders();

        std::sort(orders.begin(), orders.end(), [&origin](const Order& o1, const Order& o2) {
            return std::max(origin.distance(o1.from), origin.distance(o1.to)) < std::max(origin.distance(o2.from), origin.distance(o2.to));
            });

        auto best_sol = solve_sub(std::vector<Order>(orders.begin(), orders.begin() + M));
        //dump(best_sol.total_dist);

        auto get_temp = [](double start_temp, double end_temp, double now_time, double end_time) {
            return end_temp + (start_temp - end_temp) * (end_time - now_time) / end_time;
        };
        
        int loop = 0;
        auto prev_sol = best_sol;
        double start_time = timer.elapsed_ms(), now_time, end_time = start_time + 1900;
        while ((now_time = timer.elapsed_ms()) < end_time) {
            int i = rnd.next_int(M), j = rnd.next_int(M, N - 1);
            std::swap(orders[i], orders[j]);
            auto sol = solve_sub(std::vector<Order>(orders.begin(), orders.begin() + M));
            int diff = sol.total_dist - prev_sol.total_dist;
            double temp = get_temp(3.0, 0.0, now_time - start_time, end_time - start_time);
            double prob = (-diff / temp);
            //if (rnd.next_double() < prob) {
            if (diff < 0) {
                prev_sol = sol;
                if (sol.total_dist < best_sol.total_dist) {
                    best_sol = sol;
                    //dump(best_sol.total_dist);
                }
            }
            else {
                std::swap(orders[i], orders[j]);
            }
            loop++;
        }

        //dump(loop);

        return best_sol;
    }

};

void batch_test() {

    std::vector<int> scores(100, 0);
    concurrency::parallel_for(0, 100, [&](int seed) {
        std::string in_file = format("C:\\dev\\heuristic\\tasks\\AHC006\\tools\\in\\%04d.txt", seed);
        std::string out_file = format("C:\\dev\\heuristic\\tasks\\AHC006\\tools\\out\\%04d.txt", seed);
        std::ifstream ifs(in_file);
        std::ofstream ofs(out_file);
        TestCase tc(ifs);
        Solver solver(tc);
        auto solution = solver.solve();
        ofs << solution << std::endl;
        scores[seed] = solution.total_dist;
    });

    dump(scores);

}

int main() {
    std::ios::sync_with_stdio(false);
    std::cin.tie(nullptr);

    batch_test();
    exit(1);

#ifdef _MSC_VER
    std::ifstream ifs("C:\\dev\\heuristic\\tasks\\AHC006\\tools\\in\\0000.txt");
    std::istream& in = ifs;
    std::ofstream ofs("C:\\dev\\heuristic\\tasks\\AHC006\\tools\\out\\0000.txt");
    std::ostream& out = ofs;
#else
    std::istream& in = std::cin;
    std::ostream& out = std::cout;
#endif

    TestCase tc(in);

    Solver solver(tc);

    auto solution = solver.solve();

    out << solution << std::endl;

    return 0;
}