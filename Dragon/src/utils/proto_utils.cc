#include <cerrno>
#include <fstream>

#include <fcntl.h>

#ifdef _MSC_VER
#include <io.h>
#else
#include <unistd.h>
#endif

#include <google/protobuf/text_format.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

#include "utils/proto_utils.h"

namespace dragon {

using google::protobuf::Message;
using google::protobuf::TextFormat;
using google::protobuf::io::ArrayInputStream;
using google::protobuf::io::FileInputStream;
using google::protobuf::io::FileOutputStream;
using google::protobuf::io::CodedInputStream;
using google::protobuf::io::CodedOutputStream;
using google::protobuf::io::ZeroCopyInputStream;
using google::protobuf::io::ZeroCopyOutputStream;

bool ParseProtoFromText(
    string                      text,
    Message*                    proto) {
    return TextFormat::ParseFromString(text, proto);
}

bool ParseProtoFromLargeString(
    const string&                   str,
    Message*                        proto) {
    ArrayInputStream input_stream(str.data(), (int)str.size());
    CodedInputStream coded_stream(&input_stream);
    coded_stream.SetTotalBytesLimit(2147483647, -1);
    return proto->ParseFromCodedStream(&coded_stream);
}

bool ReadProtoFromBinaryFile(
    const char*                 filename,
    Message*                    proto) {
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
    delete coded_input; delete raw_input; close(fd);
    return success;
}

void WriteProtoToBinaryFile(
    const Message&              proto,
    const char*                 filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    CHECK_NE(fd, -1) << "File cannot be created: "
        << filename << " error number: " << errno;
    ZeroCopyOutputStream* raw_output = new FileOutputStream(fd);
    CodedOutputStream* coded_output = new CodedOutputStream(raw_output);
    CHECK(proto.SerializeToCodedStream(coded_output));
    delete coded_output; delete raw_output; close(fd);
}

}  // namespace dragon