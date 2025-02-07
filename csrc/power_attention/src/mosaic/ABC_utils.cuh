#pragma once

#include <cute/tensor.hpp>

using namespace cute;

struct A_t{}; struct B_t{}; struct C_t{};

// get_MNK
auto ABC_get_MNK(A_t, auto MNK_TileShape) {
    return make_shape(get<0>(MNK_TileShape), get<2>(MNK_TileShape));
}
auto ABC_get_MNK(B_t, auto MNK_TileShape) {
    return make_shape(get<1>(MNK_TileShape), get<2>(MNK_TileShape));
}
auto ABC_get_MNK(C_t, auto MNK_TileShape) {
    return make_shape(get<0>(MNK_TileShape), get<1>(MNK_TileShape));
}

// get_TV_layout
template<typename MmaAtom>
auto ABC_get_TV_layout(A_t, MmaAtom) {
    return typename MmaAtom::LayoutA_TV{};
}
template<typename MmaAtom>
auto ABC_get_TV_layout(B_t, MmaAtom) {
    return typename MmaAtom::LayoutB_TV{};
}
template<typename MmaAtom>
auto ABC_get_TV_layout(C_t, MmaAtom) {
    return typename MmaAtom::LayoutC_TV{};
} 