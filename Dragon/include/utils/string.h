// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_STRING_H_
#define DRAGON_UTILS_STRING_H_

#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include <cstdlib>

#include "cast.h"

namespace dragon {

inline std::vector<std::string> SplitString(const std::string& str, 
                                            const std::string& c) {
    std::vector<std::string> ret;
    std::string temp(str);
    size_t pos;
    while (pos = temp.find(c), pos != std::string::npos) {
        ret.push_back(temp.substr(0, pos));
        temp.erase(0, pos + 1);
    }
    ret.push_back(temp);
    return ret;
}

template<> inline std::string dragon_cast<std::string, int>(int val) {
    std::stringstream ss;
    ss << val;
    return ss.str();
}

template<> inline int dragon_cast<int, std::string>(std::string val) { 
    return atoi(val.c_str()); 
}

}    // namespace dragon

#endif    // DRAGON_UTILS_STRING_H_