/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_STRING_UTILS_H_
#define DRAGON_UTILS_STRING_UTILS_H_

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace dragon {

namespace str {

template <typename T>
std::string to(T val) {
  return std::to_string(val);
}

inline std::string lower(const std::string& str) {
  std::string ret(str);
  std::transform(str.begin(), str.end(), ret.begin(), ::tolower);
  return ret;
}

inline std::string upper(const std::string& str) {
  std::string ret(str);
  std::transform(str.begin(), str.end(), ret.begin(), ::toupper);
  return ret;
}

inline std::vector<std::string> split(
    const std::string& str,
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

inline bool find(const std::string& str, const std::string& pattern) {
  return str.find(pattern) != std::string::npos;
}

inline bool startswith(const std::string& str, const std::string& pattern) {
  if (pattern.size() > str.size()) return false;
  return str.compare(0, pattern.size(), pattern) == 0;
}

inline bool endswith(const std::string& str, const std::string& pattern) {
  if (pattern.size() > str.size()) return false;
  return str.compare(str.size() - pattern.size(), pattern.size(), pattern) == 0;
}

inline std::string replace_first(
    const std::string& str,
    const std::string& pattern,
    const std::string& excepted) {
  size_t pos = 0;
  if ((pos = str.find(pattern)) != std::string::npos) {
    std::string ret(str);
    ret.replace(pos, pattern.size(), excepted);
    return ret;
  } else {
    return str;
  }
}

inline std::string replace_all(
    const std::string& str,
    const std::string& pattern,
    const std::string& excepted) {
  std::string ret = str;
  size_t pos = ret.find(pattern, 0);
  while (pos != std::string::npos) {
    ret.replace(pos, pattern.size(), excepted);
    pos = ret.find(pattern, pos + excepted.size());
  }
  return ret;
}

} // namespace str

} // namespace dragon

#endif // DRAGON_UTILS_STRING_UTILS_H_
