package com.phei.netty.protocol.ten.http.xml.codec;

import io.netty.buffer.ByteBuf;
import io.netty.channel.ChannelHandlerContext;
import io.netty.handler.codec.MessageToMessageDecoder;

import java.io.StringReader;
import java.nio.charset.Charset;

import org.jibx.runtime.BindingDirectory;
import org.jibx.runtime.IBindingFactory;
import org.jibx.runtime.IUnmarshallingContext;

public abstract class AbstractHttpXmlDecoder<T> extends MessageToMessageDecoder<T> {
    private IBindingFactory factory;
    private StringReader reader;
    private Class<?> clazz;
    private boolean isPrint;
    private final static String CHARSET_NAME = "UTF-8";
    private final static Charset UTF_8 = Charset.forName(CHARSET_NAME);

    protected AbstractHttpXmlDecoder(Class<?> clazz) {
        this(clazz, false);
    }

    protected AbstractHttpXmlDecoder(Class<?> clazz, boolean isPrint) {
        this.clazz = clazz;
        this.isPrint = isPrint;
    }

    protected Object decode0(ChannelHandlerContext arg0, ByteBuf body) throws Exception {
        factory = BindingDirectory.getFactory(clazz);
        String content = body.toString(UTF_8);
        if (isPrint)
            System.out.println("The body is : " + content);
        reader = new StringReader(content);
        IUnmarshallingContext uctx = factory.createUnmarshallingContext();
        Object result = uctx.unmarshalDocument(reader);
        reader.close();
        reader = null;
        return result;
    }

    @Skip
    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) throws Exception {
        // 释放资源
        if (reader != null) {
            reader.close();
            reader = null;
        }
    }
}
