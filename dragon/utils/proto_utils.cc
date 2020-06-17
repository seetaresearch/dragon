#include <fcntl.h>
#include <cerrno>
#include <fstream>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

#include <google/protobuf/io/coded_stream.h>

#ifdef BUILD_RUNTIME
#include <google/protobuf/io/zero_copy_stream_impl_lite.h>
#else
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#endif // BUILD_RUNTIME

#include "dragon/utils/proto_utils.h"

namespace dragon {

using google::protobuf::io::CodedInputStream;

#ifdef BUILD_RUNTIME

using google::protobuf::MessageLite;
using google::protobuf::io::CopyingInputStream;
using google::protobuf::io::CopyingInputStreamAdaptor;

namespace {

// Use standard c++ input stream instead
class IfstreamInputStream : public google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const string& filename)
      : ifs_(filename.c_str(), std::ios::in | std::ios::binary) {}

  ~IfstreamInputStream() {
    ifs_.close();
  }

  int Read(void* buffer, int size) {
    if (!ifs_) return -1;
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};

} // namespace

bool ParseProtoFromText(string text, MessageLite* proto) {
  NOT_IMPLEMENTED;
  return false;
}

bool ParseProtoFromLargeString(const string& str, MessageLite* proto) {
  NOT_IMPLEMENTED;
  return false;
}

bool ReadProtoFromBinaryFile(const char* filename, MessageLite* proto) {
  CopyingInputStreamAdaptor raw_input(new IfstreamInputStream(filename));
  raw_input.SetOwnsCopyingStream(true);
  CodedInputStream coded_stream(&raw_input);
  coded_stream.SetTotalBytesLimit(2147483647, -1);
  return proto->ParseFromCodedStream(&coded_stream);
}

void WriteProtoToBinaryFile(const MessageLite& proto, const char* filename) {
  NOT_IMPLEMENTED;
}

#else

using google::protobuf::Message;
using google::protobuf::TextFormat;
using google::protobuf::io::ArrayInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

bool ParseProtoFromText(string text, Message* proto) {
  return TextFormat::ParseFromString(text, proto);
}

bool ParseProtoFromLargeString(const string& str, Message* proto) {
  ArrayInputStream input_stream(str.data(), (int)str.size());
  CodedInputStream coded_stream(&input_stream);
  coded_stream.SetTotalBytesLimit(2147483647, -1);
  return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromBinaryFile(const char* filename, Message* proto) {
#ifdef _MSC_VER
  int fd = open(filename, O_RDONLY | O_BINARY);
#else
  int fd = open(filename, O_RDONLY);
#endif
  CHECK_NE(fd, -1) << "File not found: " << filename;
  ZeroCopyInputStream* raw_input = new FileInputStream(fd);
  CodedInputStream* coded_input = new CodedInputStream(raw_input);
  coded_input->SetTotalBytesLimit(2147483647, -1);
  bool success = proto->ParseFromCodedStream(coded_input);
  delete coded_input;
  delete raw_input;
  close(fd);
  return success;
}

void WriteProtoToBinaryFile(const Message& proto, const char* filename) {
  int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
  CHECK_NE(fd, -1) << "File cannot be created: " << filename
                   << " error number: " << errno;
  ZeroCopyOutputStream* raw_output = new FileOutputStream(fd);
  CodedOutputStream* coded_output = new CodedOutputStream(raw_output);
  CHECK(proto.SerializeToCodedStream(coded_output));
  delete coded_output;
  delete raw_output;
  close(fd);
}

#endif // BUILD_RUNTIME

} // namespace dragon
