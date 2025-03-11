#pragma once

#include <iomanip>
#include <iterator>
#include <sstream>
#include <string_view>
#include <tuple>
#include <type_traits>

namespace common::to_stream_details
{

template <class T0, class T1, class = void>
struct HasStreamBitShl : std::false_type
{
};

template <class T0, class T1>
struct HasStreamBitShl<T0, T1, std::void_t<decltype(std::declval<T0>() << std::declval<T1>())>>
    : std::true_type
{
};

template <class T, class = void>
struct HasRangeBeginEnd : std::false_type
{
};

template <class T>
struct HasRangeBeginEnd<
    T,
    std::void_t<decltype(std::begin(std::declval<T>()) != std::end(std::declval<T>()))>>
    : std::true_type
{
};

template <class T, class = void>
struct HasTupleSize : std::false_type
{
};

template <class T>
struct HasTupleSize<T, std::void_t<decltype(std::tuple_size<T>::value)>> : std::true_type
{
};

struct ToStreamImpl
{
    template <class Os, class T, std::size_t... Is>
    static void
    HelperTupleToStream(Os& os, T const& t, std::string_view fms, std::index_sequence<Is...>)
    {
        os << '(';
        (ToStreamDetails(os, std::get<0>(t), fms), ...,
         (os << ' ', ToStreamDetails(os, std::get<Is + 1>(t), fms)));
        os << ')';
    }

    template <
        class Os,
        class T,
        std::enable_if_t<!HasStreamBitShl<Os&, T const&>::value && (std::tuple_size<T>::value >= 1),
                         int>
        = 0>
    static void ToStreamDetails(Os& os, T const& t, std::string_view fms)
    {
        return HelperTupleToStream(os, t, fms,
                                   std::make_index_sequence<std::tuple_size_v<T> - 1>{});
    }

    template <
        class Os,
        class T,
        std::enable_if_t<!HasStreamBitShl<Os&, T const&>::value && (std::tuple_size<T>::value == 0),
                         int>
        = 0>
    static void ToStreamDetails(Os& os, T const& t, std::string_view fms)
    {
        os << "()";
    }

    template <class Os,
              class T,
              std::enable_if_t<!HasStreamBitShl<Os&, T const&>::value && !HasTupleSize<T>::value
                                   && HasRangeBeginEnd<T>::value,
                               int>
              = 0>
    static void ToStreamDetails(Os& os, T const& t, std::string_view fms)
    {
        auto it  = std::begin(t);
        auto eit = std::end(t);
        os << '[';
        if (it != eit)
        {
            ToStreamDetails(os, *it, fms);
            ++it;
            for (; it != eit; ++it)
            {
                os << ' ';
                ToStreamDetails(os, *it, fms);
            }
        }
        os << ']';
    }

    template <
        class Os,
        class T,
        std::enable_if_t<HasStreamBitShl<Os&, T const&>::value && !std::is_enum<T>::value, int> = 0>
    static void ToStreamDetails(Os& os, T const& t, std::string_view fms)
    {
        auto flgs = os.flags();
        if (fms.size() != 0)
        {
            if (fms.size() != 0 && fms[0] == '-')
            {
                fms = fms.substr(1);
                os << std::right;
            }
            if (fms.size() != 0 && fms[0] == '0')
            {
                fms = fms.substr(1);
                os << std::setfill('0');
            }
            {
                int tmp = 0;
                while (fms.size() != 0 && '0' <= fms[0] && '9' >= fms[0])
                {
                    tmp *= 10;
                    tmp += fms[0] - '0';
                    fms = fms.substr(1);
                }
                if (tmp != 0)
                    os << std::setw(tmp);
            }
            if (fms.size() != 0 && fms[0] == '.')
            {
                fms     = fms.substr(1);
                int tmp = 0;
                while (fms.size() != 0 && '0' <= fms[0] && '9' >= fms[0])
                {
                    tmp *= 10;
                    tmp += fms[0] - '0';
                    fms = fms.substr(1);
                }
                os << std::setprecision(tmp);
            }
            if (fms.size() != 0)
            {
                switch (fms[0])
                {
                case 'x':
                    os << std::hex;
                    break;
                case 'd':
                    os << std::dec;
                    break;
                case 'o':
                    os << std::oct;
                    break;
                };
            }
        }
        os << t;
        os.flags(flgs);
    }

    template <class Os,
              class T,
              std::enable_if_t<
                  std::is_enum<T>::value
                      && HasStreamBitShl<Os&, typename std::underlying_type<T>::type const&>::value,
                  int>
              = 0>
    static void ToStreamDetails(Os& os, T const& t, std::string_view fms)
    {
        os << std::underlying_type_t<T>{t};
    }
};

}  // namespace common::to_stream_details

namespace common
{
template <class Os, class T>
void ToStreamDetails(Os& os, T const& t, std::string_view fms)
{
    to_stream_details::ToStreamImpl::ToStreamDetails(os, t, fms);
}

// 针对 C 风格一维数组的特化
template <class Os, class T, size_t N>
void ToStreamDetails(Os& os, T const (&arr)[N], std::string_view fms)
{
    os << '[';
    if (N > 0)
    {
        ToStreamDetails(os, arr[0], fms);
        for (size_t i = 1; i < N; ++i)
        {
            os << ' ';
            ToStreamDetails(os, arr[i], fms);
        }
    }
    os << ']';
}

// 针对 C 风格二维数组的特化
template <class Os, class T, size_t N, size_t M>
void ToStreamDetails(Os& os, T const (&arr)[N][M], std::string_view fms)
{
    os << '[';
    for (size_t i = 0; i < N; ++i)
    {
        if (i > 0)
            os << ' ';
        ToStreamDetails(os, arr[i], fms);  // 递归调用处理一维数组
    }
    os << ']';
}

template <class T>
static std::string to_string(T const& t, std::string_view fms)
{
    std::ostringstream ss;
    ToStreamDetails(ss, t, fms);
    return ss.str();
}

template <class T>
static std::string to_string(T const& t)
{
    if constexpr (std::is_convertible_v<T, std::string>)
    {
        return t;
    }
    else
    {
        std::ostringstream ss;
        ToStreamDetails(ss, t, {});
        return ss.str();
    }
}
}  // namespace common
