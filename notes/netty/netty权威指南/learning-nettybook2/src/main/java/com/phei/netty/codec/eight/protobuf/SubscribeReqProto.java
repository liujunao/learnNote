package com.phei.netty.codec.eight.protobuf;

import com.google.protobuf.*;
import com.google.protobuf.GeneratedMessage.*;

import java.io.*;
import java.util.Collections;
import java.util.List;

public final class SubscribeReqProto {
    public static void registerAllExtensions(ExtensionRegistry registry) {
    }

    public interface SubscribeReqOrBuilder extends MessageOrBuilder {
        boolean hasSubReqID();

        int getSubReqID();

        boolean hasUserName();

        String getUserName();

        ByteString getUserNameBytes();

        boolean hasProductName();

        String getProductName();

        ByteString getProductNameBytes();

        List<String> getAddressList();

        int getAddressCount();

        String getAddress(int index);

        ByteString getAddressBytes(int index);
    }

    public static final class SubscribeReq extends GeneratedMessage implements SubscribeReqOrBuilder {
        private SubscribeReq(GeneratedMessage.Builder<?> builder) {
            super(builder);
            this.unknownFields = builder.getUnknownFields();
        }

        private SubscribeReq(boolean noInit) {
            this.unknownFields = UnknownFieldSet.getDefaultInstance();
        }

        private static final SubscribeReq defaultInstance;

        public static SubscribeReq getDefaultInstance() {
            return defaultInstance;
        }

        public SubscribeReq getDefaultInstanceForType() {
            return defaultInstance;
        }

        private final UnknownFieldSet unknownFields;

        @Override
        public final UnknownFieldSet getUnknownFields() {
            return this.unknownFields;
        }

        private SubscribeReq(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            initFields();
            int mutable_bitField0_ = 0;
            UnknownFieldSet.Builder unknownFields = UnknownFieldSet.newBuilder();
            try {
                boolean done = false;
                while (!done) {
                    int tag = input.readTag();
                    switch (tag) {
                        case 0:
                            done = true;
                            break;
                        default: {
                            if (!parseUnknownField(input, unknownFields, extensionRegistry, tag)) {
                                done = true;
                            }
                            break;
                        }
                        case 8: {
                            bitField0_ |= 0x00000001;
                            subReqID_ = input.readInt32();
                            break;
                        }
                        case 18: {
                            bitField0_ |= 0x00000002;
                            userName_ = input.readBytes();
                            break;
                        }
                        case 26: {
                            bitField0_ |= 0x00000004;
                            productName_ = input.readBytes();
                            break;
                        }
                        case 34: {
                            if (!((mutable_bitField0_ & 0x00000008) == 0x00000008)) {
                                address_ = new LazyStringArrayList();
                                mutable_bitField0_ |= 0x00000008;
                            }
                            address_.add(input.readBytes());
                            break;
                        }
                    }
                }
            } catch (InvalidProtocolBufferException e) {
                throw e.setUnfinishedMessage(this);
            } catch (IOException e) {
                throw new InvalidProtocolBufferException(e.getMessage()).setUnfinishedMessage(this);
            } finally {
                if (((mutable_bitField0_ & 0x00000008) == 0x00000008)) {
                    address_ = new UnmodifiableLazyStringList(address_);
                }
                this.unknownFields = unknownFields.build();
                makeExtensionsImmutable();
            }
        }

        public static final Descriptors.Descriptor getDescriptor() {
            return SubscribeReqProto.internal_static_netty_SubscribeReq_descriptor;
        }

        protected FieldAccessorTable internalGetFieldAccessorTable() {
            return SubscribeReqProto.internal_static_netty_SubscribeReq_fieldAccessorTable
                    .ensureFieldAccessorsInitialized(SubscribeReqProto.SubscribeReq.class, SubscribeReqProto.SubscribeReq.Builder.class);
        }

        public static Parser<SubscribeReq> PARSER = new AbstractParser<SubscribeReq>() {
            public SubscribeReq parsePartialFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return new SubscribeReq(input, extensionRegistry);
            }
        };

        @Override
        public Parser<SubscribeReq> getParserForType() {
            return PARSER;
        }

        private int bitField0_;
        public static final int SUBREQID_FIELD_NUMBER = 1;
        private int subReqID_;

        public boolean hasSubReqID() {
            return ((bitField0_ & 0x00000001) == 0x00000001);
        }

        public int getSubReqID() {
            return subReqID_;
        }

        public static final int USERNAME_FIELD_NUMBER = 2;
        private Object userName_;

        public boolean hasUserName() {
            return ((bitField0_ & 0x00000002) == 0x00000002);
        }

        public String getUserName() {
            Object ref = userName_;
            if (ref instanceof String) {
                return (String) ref;
            } else {
                ByteString bs = (ByteString) ref;
                String s = bs.toStringUtf8();
                if (bs.isValidUtf8()) {
                    userName_ = s;
                }
                return s;
            }
        }

        public ByteString getUserNameBytes() {
            Object ref = userName_;
            if (ref instanceof String) {
                ByteString b = ByteString.copyFromUtf8((String) ref);
                userName_ = b;
                return b;
            } else {
                return (ByteString) ref;
            }
        }

        public static final int PRODUCTNAME_FIELD_NUMBER = 3;
        private Object productName_;

        public boolean hasProductName() {
            return ((bitField0_ & 0x00000004) == 0x00000004);
        }

        public String getProductName() {
            Object ref = productName_;
            if (ref instanceof String) {
                return (String) ref;
            } else {
                ByteString bs = (ByteString) ref;
                String s = bs.toStringUtf8();
                if (bs.isValidUtf8()) {
                    productName_ = s;
                }
                return s;
            }
        }

        public ByteString getProductNameBytes() {
            Object ref = productName_;
            if (ref instanceof String) {
                ByteString b = ByteString.copyFromUtf8((String) ref);
                productName_ = b;
                return b;
            } else {
                return (ByteString) ref;
            }
        }

        public static final int ADDRESS_FIELD_NUMBER = 4;
        private LazyStringList address_;

        public java.util.List<String> getAddressList() {
            return address_;
        }

        public int getAddressCount() {
            return address_.size();
        }

        public String getAddress(int index) {
            return address_.get(index);
        }

        public ByteString getAddressBytes(int index) {
            return address_.getByteString(index);
        }

        private void initFields() {
            subReqID_ = 0;
            userName_ = "";
            productName_ = "";
            address_ = LazyStringArrayList.EMPTY;
        }

        private byte memoizedIsInitialized = -1;

        public final boolean isInitialized() {
            byte isInitialized = memoizedIsInitialized;
            if (isInitialized != -1) return isInitialized == 1;

            if (!hasSubReqID()) {
                memoizedIsInitialized = 0;
                return false;
            }
            if (!hasUserName()) {
                memoizedIsInitialized = 0;
                return false;
            }
            if (!hasProductName()) {
                memoizedIsInitialized = 0;
                return false;
            }
            memoizedIsInitialized = 1;
            return true;
        }

        public void writeTo(CodedOutputStream output) throws IOException {
            getSerializedSize();
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
                output.writeInt32(1, subReqID_);
            }
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
                output.writeBytes(2, getUserNameBytes());
            }
            if (((bitField0_ & 0x00000004) == 0x00000004)) {
                output.writeBytes(3, getProductNameBytes());
            }
            for (int i = 0; i < address_.size(); i++) {
                output.writeBytes(4, address_.getByteString(i));
            }
            getUnknownFields().writeTo(output);
        }

        private int memoizedSerializedSize = -1;

        public int getSerializedSize() {
            int size = memoizedSerializedSize;
            if (size != -1) return size;
            size = 0;
            if (((bitField0_ & 0x00000001) == 0x00000001)) {
                size += CodedOutputStream.computeInt32Size(1, subReqID_);
            }
            if (((bitField0_ & 0x00000002) == 0x00000002)) {
                size += CodedOutputStream.computeBytesSize(2, getUserNameBytes());
            }
            if (((bitField0_ & 0x00000004) == 0x00000004)) {
                size += CodedOutputStream.computeBytesSize(3, getProductNameBytes());
            }
            {
                int dataSize = 0;
                for (int i = 0; i < address_.size(); i++) {
                    dataSize += CodedOutputStream.computeBytesSizeNoTag(address_.getByteString(i));
                }
                size += dataSize;
                size += 1 * getAddressList().size();
            }
            size += getUnknownFields().getSerializedSize();
            memoizedSerializedSize = size;
            return size;
        }

        private static final long serialVersionUID = 0L;

        @Override
        protected Object writeReplace() throws ObjectStreamException {
            return super.writeReplace();
        }

        public static SubscribeReq parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data);
        }

        public static SubscribeReq parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data, extensionRegistry);
        }

        public static SubscribeReq parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data);
        }

        public static SubscribeReq parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data, extensionRegistry);
        }

        public static SubscribeReq parseFrom(InputStream input) throws IOException {
            return PARSER.parseFrom(input);
        }

        public static SubscribeReq parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return PARSER.parseFrom(input, extensionRegistry);
        }

        public static SubscribeReq parseDelimitedFrom(InputStream input) throws IOException {
            return PARSER.parseDelimitedFrom(input);
        }

        public static SubscribeReq parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return PARSER.parseDelimitedFrom(input, extensionRegistry);
        }

        public static SubscribeReq parseFrom(CodedInputStream input) throws IOException {
            return PARSER.parseFrom(input);
        }

        public static SubscribeReq parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return PARSER.parseFrom(input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return Builder.create();
        }

        public Builder newBuilderForType() {
            return newBuilder();
        }

        public static Builder newBuilder(SubscribeReq prototype) {
            return newBuilder().mergeFrom(prototype);
        }

        public Builder toBuilder() {
            return newBuilder(this);
        }

        @Override
        protected Builder newBuilderForType(BuilderParent parent) {
            Builder builder = new Builder(parent);
            return builder;
        }

        public static final class Builder extends GeneratedMessage.Builder<Builder> implements SubscribeReqOrBuilder {
            public static final Descriptors.Descriptor
            getDescriptor() {
                return SubscribeReqProto.internal_static_netty_SubscribeReq_descriptor;
            }

            protected GeneratedMessage.FieldAccessorTable
            internalGetFieldAccessorTable() {
                return SubscribeReqProto.internal_static_netty_SubscribeReq_fieldAccessorTable
                        .ensureFieldAccessorsInitialized(SubscribeReq.class, Builder.class);
            }

            private Builder() {
                maybeForceBuilderInitialization();
            }

            private Builder(GeneratedMessage.BuilderParent parent) {
                super(parent);
                maybeForceBuilderInitialization();
            }

            private void maybeForceBuilderInitialization() {
                if (GeneratedMessage.alwaysUseFieldBuilders) {
                }
            }

            private static Builder create() {
                return new Builder();
            }

            public Builder clear() {
                super.clear();
                subReqID_ = 0;
                bitField0_ = (bitField0_ & ~0x00000001);
                userName_ = "";
                bitField0_ = (bitField0_ & ~0x00000002);
                productName_ = "";
                bitField0_ = (bitField0_ & ~0x00000004);
                address_ = LazyStringArrayList.EMPTY;
                bitField0_ = (bitField0_ & ~0x00000008);
                return this;
            }

            public Builder clone() {
                return create().mergeFrom(buildPartial());
            }

            public Descriptors.Descriptor
            getDescriptorForType() {
                return SubscribeReqProto.internal_static_netty_SubscribeReq_descriptor;
            }

            public SubscribeReq getDefaultInstanceForType() {
                return SubscribeReq.getDefaultInstance();
            }

            public SubscribeReq build() {
                SubscribeReq result = buildPartial();
                if (!result.isInitialized()) {
                    throw newUninitializedMessageException(result);
                }
                return result;
            }

            public SubscribeReq buildPartial() {
                SubscribeReq result = new SubscribeReq(this);
                int from_bitField0_ = bitField0_;
                int to_bitField0_ = 0;
                if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
                    to_bitField0_ |= 0x00000001;
                }
                result.subReqID_ = subReqID_;
                if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
                    to_bitField0_ |= 0x00000002;
                }
                result.userName_ = userName_;
                if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
                    to_bitField0_ |= 0x00000004;
                }
                result.productName_ = productName_;
                if (((bitField0_ & 0x00000008) == 0x00000008)) {
                    address_ = new UnmodifiableLazyStringList(
                            address_);
                    bitField0_ = (bitField0_ & ~0x00000008);
                }
                result.address_ = address_;
                result.bitField0_ = to_bitField0_;
                onBuilt();
                return result;
            }

            public Builder mergeFrom(Message other) {
                if (other instanceof SubscribeReq) {
                    return mergeFrom((SubscribeReq) other);
                } else {
                    super.mergeFrom(other);
                    return this;
                }
            }

            public Builder mergeFrom(SubscribeReq other) {
                if (other == SubscribeReq.getDefaultInstance())
                    return this;
                if (other.hasSubReqID()) {
                    setSubReqID(other.getSubReqID());
                }
                if (other.hasUserName()) {
                    bitField0_ |= 0x00000002;
                    userName_ = other.userName_;
                    onChanged();
                }
                if (other.hasProductName()) {
                    bitField0_ |= 0x00000004;
                    productName_ = other.productName_;
                    onChanged();
                }
                if (!other.address_.isEmpty()) {
                    if (address_.isEmpty()) {
                        address_ = other.address_;
                        bitField0_ = (bitField0_ & ~0x00000008);
                    } else {
                        ensureAddressIsMutable();
                        address_.addAll(other.address_);
                    }
                    onChanged();
                }
                this.mergeUnknownFields(other.getUnknownFields());
                return this;
            }

            public final boolean isInitialized() {
                if (!hasSubReqID()) {
                    return false;
                }
                if (!hasUserName()) {
                    return false;
                }
                if (!hasProductName()) {
                    return false;
                }
                return true;
            }

            public Builder mergeFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                SubscribeReq parsedMessage = null;
                try {
                    parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
                } catch (InvalidProtocolBufferException e) {
                    parsedMessage = (SubscribeReq) e.getUnfinishedMessage();
                    throw e;
                } finally {
                    if (parsedMessage != null) {
                        mergeFrom(parsedMessage);
                    }
                }
                return this;
            }

            private int bitField0_;
            private int subReqID_;

            public boolean hasSubReqID() {
                return ((bitField0_ & 0x00000001) == 0x00000001);
            }

            public int getSubReqID() {
                return subReqID_;
            }

            public Builder setSubReqID(int value) {
                bitField0_ |= 0x00000001;
                subReqID_ = value;
                onChanged();
                return this;
            }

            public Builder clearSubReqID() {
                bitField0_ = (bitField0_ & ~0x00000001);
                subReqID_ = 0;
                onChanged();
                return this;
            }

            private Object userName_ = "";

            public boolean hasUserName() {
                return ((bitField0_ & 0x00000002) == 0x00000002);
            }

            public String getUserName() {
                Object ref = userName_;
                if (!(ref instanceof String)) {
                    String s = ((ByteString) ref).toStringUtf8();
                    userName_ = s;
                    return s;
                } else {
                    return (String) ref;
                }
            }

            public ByteString getUserNameBytes() {
                Object ref = userName_;
                if (ref instanceof String) {
                    ByteString b = ByteString.copyFromUtf8((String) ref);
                    userName_ = b;
                    return b;
                } else {
                    return (ByteString) ref;
                }
            }

            public Builder setUserName(String value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                bitField0_ |= 0x00000002;
                userName_ = value;
                onChanged();
                return this;
            }

            public Builder clearUserName() {
                bitField0_ = (bitField0_ & ~0x00000002);
                userName_ = getDefaultInstance().getUserName();
                onChanged();
                return this;
            }

            public Builder setUserNameBytes(ByteString value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                bitField0_ |= 0x00000002;
                userName_ = value;
                onChanged();
                return this;
            }

            private Object productName_ = "";

            public boolean hasProductName() {
                return ((bitField0_ & 0x00000004) == 0x00000004);
            }

            public String getProductName() {
                Object ref = productName_;
                if (!(ref instanceof String)) {
                    String s = ((ByteString) ref).toStringUtf8();
                    productName_ = s;
                    return s;
                } else {
                    return (String) ref;
                }
            }

            public ByteString getProductNameBytes() {
                Object ref = productName_;
                if (ref instanceof String) {
                    ByteString b = ByteString.copyFromUtf8((String) ref);
                    productName_ = b;
                    return b;
                } else {
                    return (ByteString) ref;
                }
            }

            public Builder setProductName(String value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                bitField0_ |= 0x00000004;
                productName_ = value;
                onChanged();
                return this;
            }

            public Builder clearProductName() {
                bitField0_ = (bitField0_ & ~0x00000004);
                productName_ = getDefaultInstance().getProductName();
                onChanged();
                return this;
            }

            public Builder setProductNameBytes(ByteString value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                bitField0_ |= 0x00000004;
                productName_ = value;
                onChanged();
                return this;
            }

            private LazyStringList address_ = LazyStringArrayList.EMPTY;

            private void ensureAddressIsMutable() {
                if (!((bitField0_ & 0x00000008) == 0x00000008)) {
                    address_ = new LazyStringArrayList(address_);
                    bitField0_ |= 0x00000008;
                }
            }

            public java.util.List<String> getAddressList() {
                return Collections.unmodifiableList(address_);
            }

            public int getAddressCount() {
                return address_.size();
            }

            public String getAddress(int index) {
                return address_.get(index);
            }

            public ByteString getAddressBytes(int index) {
                return address_.getByteString(index);
            }

            public Builder setAddress(int index, String value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                ensureAddressIsMutable();
                address_.set(index, value);
                onChanged();
                return this;
            }

            public Builder addAddress(String value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                ensureAddressIsMutable();
                address_.add(value);
                onChanged();
                return this;
            }

            public Builder addAllAddress(Iterable<String> values) {
                ensureAddressIsMutable();
                super.addAll(values, address_);
                onChanged();
                return this;
            }

            public Builder clearAddress() {
                address_ = LazyStringArrayList.EMPTY;
                bitField0_ = (bitField0_ & ~0x00000008);
                onChanged();
                return this;
            }

            public Builder addAddressBytes(ByteString value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                ensureAddressIsMutable();
                address_.add(value);
                onChanged();
                return this;
            }
        }

        static {
            defaultInstance = new SubscribeReq(true);
            defaultInstance.initFields();
        }
    }

    private static Descriptors.Descriptor internal_static_netty_SubscribeReq_descriptor;
    private static FieldAccessorTable internal_static_netty_SubscribeReq_fieldAccessorTable;

    public static Descriptors.FileDescriptor getDescriptor() {
        return descriptor;
    }

    private static Descriptors.FileDescriptor descriptor;

    static {
        String[] descriptorData = {
                "\n\030netty/SubscribeReq.proto\022\005netty\"X\n\014Sub" +
                        "scribeReq\022\020\n\010subReqID\030\001 \002(\005\022\020\n\010userName\030" +
                        "\002 \002(\t\022\023\n\013productName\030\003 \002(\t\022\017\n\007address\030\004 " +
                        "\003(\tB2\n\035com.phei.netty.codec.protobufB\021Su" +
                        "bscribeReqProto"
        };
        Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
                new Descriptors.FileDescriptor.InternalDescriptorAssigner() {
                    public ExtensionRegistry assignDescriptors(Descriptors.FileDescriptor root) {
                        descriptor = root;
                        internal_static_netty_SubscribeReq_descriptor = getDescriptor().getMessageTypes().get(0);
                        internal_static_netty_SubscribeReq_fieldAccessorTable = new GeneratedMessage.FieldAccessorTable(
                                internal_static_netty_SubscribeReq_descriptor,
                                new String[]{"SubReqID", "UserName", "ProductName", "Address",});
                        return null;
                    }
                };
        Descriptors.FileDescriptor.internalBuildGeneratedFileFrom(descriptorData, new Descriptors.FileDescriptor[]{}, assigner);
    }
}
