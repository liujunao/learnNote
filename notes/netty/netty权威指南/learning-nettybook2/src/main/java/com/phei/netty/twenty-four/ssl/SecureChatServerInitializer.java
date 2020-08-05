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

public class SecureChatServerInitializer extends ChannelInitializer<SocketChannel> {
    private String tlsMode;

    public SecureChatServerInitializer(String TLS_MODE) {
        tlsMode = TLS_MODE;
    }

    @Override
    public void initChannel(SocketChannel ch) throws Exception {
        ChannelPipeline pipeline = ch.pipeline();
        SSLEngine engine = null;
        if (com.phei.netty.ssl.SSLMODE.CA.toString().equals(tlsMode)) {
            engine = com.phei.netty.ssl.SecureChatSslContextFactory.getServerContext(tlsMode,
                    System.getProperty("user.dir") + "/src/com/phei/netty/ssl/conf/client/sChat.jks", null).createSSLEngine();
        } else if (com.phei.netty.ssl.SSLMODE.CSA.toString().equals(tlsMode)) {
            engine = com.phei.netty.ssl.SecureChatSslContextFactory.getServerContext(tlsMode,
                    System.getProperty("user.dir") + "/src/com/phei/netty/ssl/conf/twoway/sChat.jks",
                    System.getProperty("user.dir") + "/src/com/phei/netty/ssl/conf/twoway/sChat.jks").createSSLEngine();
        } else {
            System.err.println("ERROR : " + tlsMode);
            System.exit(-1);
        }
        engine.setUseClientMode(false);
        if (com.phei.netty.ssl.SSLMODE.CSA.toString().equals(tlsMode))
            engine.setNeedClientAuth(true);
        pipeline.addLast("ssl", new SslHandler(engine));
        pipeline.addLast("framer", new DelimiterBasedFrameDecoder(8192, Delimiters.lineDelimiter()));
        pipeline.addLast("decoder", new StringDecoder());
        pipeline.addLast("encoder", new StringEncoder());
        pipeline.addLast("handler", new com.phei.netty.ssl.SecureChatServerHandler());
    }
}
