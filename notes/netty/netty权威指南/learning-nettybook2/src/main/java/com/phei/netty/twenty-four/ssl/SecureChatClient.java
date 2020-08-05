package com.phei.netty.ssl;

import io.netty.bootstrap.Bootstrap;
import io.netty.channel.Channel;
import io.netty.channel.ChannelFuture;
import io.netty.channel.EventLoopGroup;
import io.netty.channel.nio.NioEventLoopGroup;
import io.netty.channel.socket.nio.NioSocketChannel;

import java.io.BufferedReader;
import java.io.InputStreamReader;

public class SecureChatClient {
    private final String host;
    private final int port;
    private final String sslMode;

    public SecureChatClient(String host, int port, String sslMode) {
        this.host = host;
        this.port = port;
        this.sslMode = sslMode;
    }

    public void run() throws Exception {
        EventLoopGroup group = new NioEventLoopGroup();
        try {
            Bootstrap b = new Bootstrap();
            b.group(group)
                    .channel(NioSocketChannel.class)
                    .handler(new com.phei.netty.ssl.SecureChatClientInitializer(sslMode));
            Channel ch = b.connect(host, port).sync().channel();
            ChannelFuture lastWriteFuture = null;
            BufferedReader in = new BufferedReader(new InputStreamReader(System.in));
            for (; ; ) {
                String line = in.readLine();
                if (line == null) {
                    break;
                }
                lastWriteFuture = ch.writeAndFlush(line + "\r\n");
                if ("bye".equals(line.toLowerCase())) {
                    ch.closeFuture().sync();
                    break;
                }
            }
            if (lastWriteFuture != null) {
                lastWriteFuture.sync();
            }
        } finally {
            group.shutdownGracefully();
        }
    }

    public static void main(String[] args) throws Exception {
        if (args.length != 1) {
            System.err.println("Usage: " + SecureChatClient.class.getSimpleName() + " <sslmode>");
            return;
        }
        String sslMode = args[0];
        new SecureChatClient("localhost", 8443, sslMode).run();
    }
}
