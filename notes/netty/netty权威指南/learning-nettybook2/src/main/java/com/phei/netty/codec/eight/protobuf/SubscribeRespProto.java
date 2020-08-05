package com.phei.netty.codec.eight.protobuf;

import com.google.protobuf.*;

import java.io.*;

public final class SubscribeRespProto {
    public static void registerAllExtensions(
            ExtensionRegistry registry) {
    }

    public interface SubscribeRespOrBuilder extends MessageOrBuilder {
        boolean hasSubReqID();

        int getSubReqID();

        boolean hasRespCode();

        int getRespCode();

        boolean hasDesc();

        String getDesc();

        ByteString getDescBytes();
    }

    public static final class SubscribeResp extends GeneratedMessage implements SubscribeRespOrBuilder {
        private SubscribeResp(GeneratedMessage.Builder<?> builder) {
            super(builder);
            this.unknownFields = builder.getUnknownFields();
        }

        private SubscribeResp(boolean noInit) {
            this.unknownFields = UnknownFieldSet.getDefaultInstance();
        }

        private static final SubscribeResp defaultInstance;

        public static SubscribeResp getDefaultInstance() {
            return defaultInstance;
        }

        public SubscribeResp getDefaultInstanceForType() {
            return defaultInstance;
        }

        private final UnknownFieldSet unknownFields;

        @Override
        public final UnknownFieldSet getUnknownFields() {
            return this.unknownFields;
        }

        private SubscribeResp(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
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
                        case 16: {
                            bitField0_ |= 0x00000002;
                            respCode_ = input.readInt32();
                            break;
                        }
                        case 26: {
                            bitField0_ |= 0x00000004;
                            desc_ = input.readBytes();
                            break;
                        }
                    }
                }
            } catch (InvalidProtocolBufferException e) {
                throw e.setUnfinishedMessage(this);
            } catch (IOException e) {
                throw new InvalidProtocolBufferException(e.getMessage()).setUnfinishedMessage(this);
            } finally {
                this.unknownFields = unknownFields.build();
                makeExtensionsImmutable();
            }
        }

        public static final Descriptors.Descriptor getDescriptor() {
            return SubscribeRespProto.internal_static_netty_SubscribeResp_descriptor;
        }

        protected GeneratedMessage.FieldAccessorTable
        internalGetFieldAccessorTable() {
            return SubscribeRespProto.internal_static_netty_SubscribeResp_fieldAccessorTable
                    .ensureFieldAccessorsInitialized(SubscribeResp.class, SubscribeResp.Builder.class);
        }

        public static Parser<SubscribeResp> PARSER = new AbstractParser<SubscribeResp>() {
            public SubscribeResp parsePartialFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
                return new SubscribeResp(input, extensionRegistry);
            }
        };

        @Override
        public Parser<SubscribeResp> getParserForType() {
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

        public static final int RESPCODE_FIELD_NUMBER = 2;
        private int respCode_;

        public boolean hasRespCode() {
            return ((bitField0_ & 0x00000002) == 0x00000002);
        }

        public int getRespCode() {
            return respCode_;
        }

        public static final int DESC_FIELD_NUMBER = 3;
        private Object desc_;

        public boolean hasDesc() {
            return ((bitField0_ & 0x00000004) == 0x00000004);
        }

        public String getDesc() {
            Object ref = desc_;
            if (ref instanceof String) {
                return (String) ref;
            } else {
                ByteString bs = (ByteString) ref;
                String s = bs.toStringUtf8();
                if (bs.isValidUtf8()) {
                    desc_ = s;
                }
                return s;
            }
        }

        public ByteString
        getDescBytes() {
            Object ref = desc_;
            if (ref instanceof String) {
                ByteString b = ByteString.copyFromUtf8((String) ref);
                desc_ = b;
                return b;
            } else {
                return (ByteString) ref;
            }
        }

        private void initFields() {
            subReqID_ = 0;
            respCode_ = 0;
            desc_ = "";
        }

        private byte memoizedIsInitialized = -1;

        public final boolean isInitialized() {
            byte isInitialized = memoizedIsInitialized;
            if (isInitialized != -1)
                return isInitialized == 1;
            if (!hasSubReqID()) {
                memoizedIsInitialized = 0;
                return false;
            }
            if (!hasRespCode()) {
                memoizedIsInitialized = 0;
                return false;
            }
            if (!hasDesc()) {
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
                output.writeInt32(2, respCode_);
            }
            if (((bitField0_ & 0x00000004) == 0x00000004)) {
                output.writeBytes(3, getDescBytes());
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
                size += CodedOutputStream.computeInt32Size(2, respCode_);
            }
            if (((bitField0_ & 0x00000004) == 0x00000004)) {
                size += CodedOutputStream.computeBytesSize(3, getDescBytes());
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

        public static SubscribeResp parseFrom(ByteString data) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data);
        }

        public static SubscribeResp parseFrom(ByteString data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data, extensionRegistry);
        }

        public static SubscribeResp parseFrom(byte[] data) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data);
        }

        public static SubscribeResp parseFrom(byte[] data, ExtensionRegistryLite extensionRegistry) throws InvalidProtocolBufferException {
            return PARSER.parseFrom(data, extensionRegistry);
        }

        public static SubscribeResp parseFrom(InputStream input) throws IOException {
            return PARSER.parseFrom(input);
        }

        public static SubscribeResp parseFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return PARSER.parseFrom(input, extensionRegistry);
        }

        public static SubscribeResp parseDelimitedFrom(InputStream input) throws IOException {
            return PARSER.parseDelimitedFrom(input);
        }

        public static SubscribeResp parseDelimitedFrom(InputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return PARSER.parseDelimitedFrom(input, extensionRegistry);
        }

        public static SubscribeResp parseFrom(CodedInputStream input) throws IOException {
            return PARSER.parseFrom(input);
        }

        public static SubscribeResp parseFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
            return PARSER.parseFrom(input, extensionRegistry);
        }

        public static Builder newBuilder() {
            return Builder.create();
        }

        public Builder newBuilderForType() {
            return newBuilder();
        }

        public static Builder newBuilder(SubscribeResp prototype) {
            return newBuilder().mergeFrom(prototype);
        }

        public Builder toBuilder() {
            return newBuilder(this);
        }

        @Override
        protected Builder newBuilderForType(GeneratedMessage.BuilderParent parent) {
            Builder builder = new Builder(parent);
            return builder;
        }

        public static final class Builder extends GeneratedMessage.Builder<Builder> implements SubscribeRespOrBuilder {
            public static final Descriptors.Descriptor
            getDescriptor() {
                return SubscribeRespProto.internal_static_netty_SubscribeResp_descriptor;
            }

            protected GeneratedMessage.FieldAccessorTable
            internalGetFieldAccessorTable() {
                return SubscribeRespProto.internal_static_netty_SubscribeResp_fieldAccessorTable
                        .ensureFieldAccessorsInitialized(SubscribeResp.class, SubscribeResp.Builder.class);
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
                respCode_ = 0;
                bitField0_ = (bitField0_ & ~0x00000002);
                desc_ = "";
                bitField0_ = (bitField0_ & ~0x00000004);
                return this;
            }

            public Builder clone() {
                return create().mergeFrom(buildPartial());
            }

            public Descriptors.Descriptor getDescriptorForType() {
                return SubscribeRespProto.internal_static_netty_SubscribeResp_descriptor;
            }

            public SubscribeResp getDefaultInstanceForType() {
                return SubscribeResp.getDefaultInstance();
            }

            public SubscribeResp build() {
                SubscribeResp result = buildPartial();
                if (!result.isInitialized()) {
                    throw newUninitializedMessageException(result);
                }
                return result;
            }

            public SubscribeResp buildPartial() {
                SubscribeResp result = new SubscribeResp(this);
                int from_bitField0_ = bitField0_;
                int to_bitField0_ = 0;
                if (((from_bitField0_ & 0x00000001) == 0x00000001)) {
                    to_bitField0_ |= 0x00000001;
                }
                result.subReqID_ = subReqID_;
                if (((from_bitField0_ & 0x00000002) == 0x00000002)) {
                    to_bitField0_ |= 0x00000002;
                }
                result.respCode_ = respCode_;
                if (((from_bitField0_ & 0x00000004) == 0x00000004)) {
                    to_bitField0_ |= 0x00000004;
                }
                result.desc_ = desc_;
                result.bitField0_ = to_bitField0_;
                onBuilt();
                return result;
            }

            public Builder mergeFrom(Message other) {
                if (other instanceof SubscribeResp) {
                    return mergeFrom((SubscribeResp) other);
                } else {
                    super.mergeFrom(other);
                    return this;
                }
            }

            public Builder mergeFrom(SubscribeResp other) {
                if (other == SubscribeResp.getDefaultInstance())
                    return this;
                if (other.hasSubReqID()) {
                    setSubReqID(other.getSubReqID());
                }
                if (other.hasRespCode()) {
                    setRespCode(other.getRespCode());
                }
                if (other.hasDesc()) {
                    bitField0_ |= 0x00000004;
                    desc_ = other.desc_;
                    onChanged();
                }
                this.mergeUnknownFields(other.getUnknownFields());
                return this;
            }

            public final boolean isInitialized() {
                if (!hasSubReqID())
                    return false;
                if (!hasRespCode())
                    return false;
                if (!hasDesc())
                    return false;
                return true;
            }

            public Builder mergeFrom(CodedInputStream input, ExtensionRegistryLite extensionRegistry) throws IOException {
                SubscribeResp parsedMessage = null;
                try {
                    parsedMessage = PARSER.parsePartialFrom(input, extensionRegistry);
                } catch (InvalidProtocolBufferException e) {
                    parsedMessage = (SubscribeResp) e.getUnfinishedMessage();
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

            private int respCode_;

            public boolean hasRespCode() {
                return ((bitField0_ & 0x00000002) == 0x00000002);
            }

            public int getRespCode() {
                return respCode_;
            }

            public Builder setRespCode(int value) {
                bitField0_ |= 0x00000002;
                respCode_ = value;
                onChanged();
                return this;
            }

            public Builder clearRespCode() {
                bitField0_ = (bitField0_ & ~0x00000002);
                respCode_ = 0;
                onChanged();
                return this;
            }

            private Object desc_ = "";

            public boolean hasDesc() {
                return ((bitField0_ & 0x00000004) == 0x00000004);
            }

            public String getDesc() {
                Object ref = desc_;
                if (!(ref instanceof String)) {
                    String s = ((ByteString) ref).toStringUtf8();
                    desc_ = s;
                    return s;
                } else {
                    return (String) ref;
                }
            }

            public ByteString getDescBytes() {
                Object ref = desc_;
                if (ref instanceof String) {
                    ByteString b = ByteString.copyFromUtf8((String) ref);
                    desc_ = b;
                    return b;
                } else {
                    return (ByteString) ref;
                }
            }

            public Builder setDesc(String value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                bitField0_ |= 0x00000004;
                desc_ = value;
                onChanged();
                return this;
            }

            public Builder clearDesc() {
                bitField0_ = (bitField0_ & ~0x00000004);
                desc_ = getDefaultInstance().getDesc();
                onChanged();
                return this;
            }

            public Builder setDescBytes(ByteString value) {
                if (value == null) {
                    throw new NullPointerException();
                }
                bitField0_ |= 0x00000004;
                desc_ = value;
                onChanged();
                return this;
            }
        }

        static {
            defaultInstance = new SubscribeResp(true);
            defaultInstance.initFields();
        }
    }

    private static Descriptors.Descriptor internal_static_netty_SubscribeResp_descriptor;
    private static GeneratedMessage.FieldAccessorTable internal_static_netty_SubscribeResp_fieldAccessorTable;

    public static Descriptors.FileDescriptor getDescriptor() {
        return descriptor;
    }

    private static Descriptors.FileDescriptor descriptor;

    static {
        String[] descriptorData = {
                "\n\031netty/SubscribeResp.proto\022\005netty\"A\n\rSu" +
                        "bscribeResp\022\020\n\010subReqID\030\001 \002(\005\022\020\n\010respCod" +
                        "e\030\002 \002(\005\022\014\n\004desc\030\003 \002(\tB3\n\035com.phei.netty." +
                        "codec.protobufB\022SubscribeRespProto"
        };
        Descriptors.FileDescriptor.InternalDescriptorAssigner assigner =
                new Descriptors.FileDescriptor.InternalDescriptorAssigner() {
                    public ExtensionRegistry assignDescriptors(Descriptors.FileDescriptor root) {
                        descriptor = root;
                        internal_static_netty_SubscribeResp_descriptor = getDescriptor().getMessageTypes().get(0);
                        internal_static_netty_SubscribeResp_fieldAccessorTable = new GeneratedMessage.FieldAccessorTable(internal_static_netty_SubscribeResp_descriptor, new String[]{"SubReqID", "RespCode", "Desc",});
                        return null;
                    }
                };
        Descriptors.FileDescriptor.internalBuildGeneratedFileFrom(descriptorData, new Descriptors.FileDescriptor[]{}, assigner);
    }
}