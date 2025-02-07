namespace power_attention
{
namespace mosaic
{
using namespace cute;

constexpr size_t n_choose_k(size_t n, size_t k) {
    if (k > n) return 0;
    if (k == 0) return 1;
    size_t result = 1;
    for(size_t i = 1; i <= k; ++i) {
        result *= (n - i + 1);
        result /= i;
    }
    return result;
}
constexpr size_t factorial(size_t n) {
    if (n <= 1) return 1;
    return n * factorial(n - 1);
}

template<int... Is> constexpr
auto dynamic_zero_tuple(cute::int_sequence<Is...>) {
    auto return_zero = [](int x) { return 0; };
    return cute::tuple(return_zero(Is)...);
}
template<int rank>
auto make_tuple_of_rank() {
    return dynamic_zero_tuple(cute::make_int_sequence<rank>());
}
template<int _rng, int _len>
struct NonDecSeq {
    static constexpr int rng = _rng;
    static constexpr int len = _len;
    static constexpr int num_elements = n_choose_k(rng + len - 1, len);
    using current_t = decltype(make_tuple_of_rank<len>());
    current_t current;
    int idx = 0;
    CUTE_HOST_DEVICE
    NonDecSeq& operator++() {
        increment<len-1>();
        idx++;
        return *this;
    }
    CUTE_HOST_DEVICE
    NonDecSeq operator+(int n) {
        NonDecSeq<rng, len> result = *this;
        for (int i = 0; i < n; ++i) {
            ++result;
        }
        return result;
    }
    template<int r> CUTE_HOST_DEVICE
    void increment() {
        if (get<r>(current) < rng-1) {
            get<r>(current)++;
        }
        else if constexpr (r > 0) {
            increment<r-1>();
            get<r>(current) = get<r-1>(current);
        }
    }
    CUTE_HOST_DEVICE
    int duplicate_count() {
        // auto arr = to_array<int>(current);
        // int hist[rng] = {}; // zero initialize the array
        // for(int i = 0; i < len; ++i) {
        //     hist[arr[i]]++;
        // }
        // int result = factorial(rng);
        // for (int i = 0; i < rng; ++i) {
        //     result /= factorial(hist[i]);
        // }
        // return result;
        static_assert(len==2);
        if (get<0>(current) == get<1>(current)) {
            return 1;
        }
        return 2;
    }
};

} // namespace mosaic
} // namespace power_attention


// rng=3
// seq = 0 0 1   // count0 = 2, count1 = 1, count2=0
// duplicates = 3! / (2! * 1!)


/* 0 -> 64
0!
1!
2!

factorials=(0!...len!)
*/

