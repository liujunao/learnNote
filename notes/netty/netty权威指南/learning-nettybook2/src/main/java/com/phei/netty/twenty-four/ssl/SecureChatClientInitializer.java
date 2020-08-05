package com.phei.netty.ssl;

import io.netty.channel.ChannelInitializer;
import io.netty.channel.ChannelPipeline;
import io.netty.channel.socket.SocketChannel;
import io.netty.handler.codec.DelimiterBasedFrameDecoder;
import io.netty.handler.codec.Delimiters;
import io.netty.handler.codec.string.StringDecoder;
import io.netty.handler.codec.string.StringEncoder;
import io.netty.handler.ssl.SslHandler;

import javax.net.ssl.SSLEngine;

public class SecureChatClientInitializer extends ChannelInitializer<SocketChannel> {
    private String tlsMode;

    public SecureChatClientInitializer(String tlsMode) {
        this.tlsMode = tlsMode;
    }

    @Override
    public void initChannel(SocketChannel ch) throws Exception {
        ChannelPipeline pipeline = ch.pipeline();
        SSLEngine engine = null;
        if (com.phei.netty.ssl.SSLMODE.CA.toString().equals(tlsMode)) {
            engine = com.phei.netty.ssl.SecureChatSslContextFactory.getClientContext(tlsMode, null,
                    System.getProperty("user.dir") + "/src/com/phei/netty/ssl/conf/client/cChat.jks").createSSLEngine();
        } else if (com.phei.netty.ssl.SSLMODE.CSA.toString().equals(tlsMode)) {
            engine = com.phei.netty.ssl.SecureChatSslContextFactory.getClientContext(tlsMode,
                    System.getProperty("user.dir") + "/src/com/phei/netty/ssl/conf/twoway/cChat.jks",
                    System.getProperty("user.dir") + "/src/com/phei/netty/ssl/conf/twoway/cChat.jks").createSSLEngine();
        } else {
            System.err.println("ERROR : " + tlsMode);
            System.exit(-1);
        }
        engine.setUseClientMode(true);
        pipeline.addLast("ssl", new SslHandler(engine));

        pipeline.addLast("framer", new DelimiterBasedFrameDecoder(8192, Delimiters.lineDelimiter()));
        pipeline.addLast("decoder", new StringDecoder());
        pipeline.addLast("encoder", new StringEncoder());
        pipeline.addLast("handler", new com.phei.netty.ssl.SecureChatClientHandler());
    }
}
