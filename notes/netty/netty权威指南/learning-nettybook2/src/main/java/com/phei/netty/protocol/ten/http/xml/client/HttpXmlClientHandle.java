package com.phei.netty.protocol.ten.http.xml.client;

import io.netty.channel.ChannelHandlerContext;
import io.netty.channel.SimpleChannelInboundHandler;

import com.phei.netty.protocol.ten.http.xml.codec.HttpXmlRequest;
import com.phei.netty.protocol.ten.http.xml.codec.HttpXmlResponse;
import com.phei.netty.protocol.ten.http.xml.pojo.OrderFactory;

public class HttpXmlClientHandle extends SimpleChannelInboundHandler<HttpXmlResponse> {
    @Override
    public void channelActive(ChannelHandlerContext ctx) {
        HttpXmlRequest request = new HttpXmlRequest(null, OrderFactory.create(123));
        ctx.writeAndFlush(request);
    }

    @Override
    public void exceptionCaught(ChannelHandlerContext ctx, Throwable cause) {
        cause.printStackTrace();
        ctx.close();
    }

    @Override
    protected void messageReceived(ChannelHandlerContext ctx, HttpXmlResponse msg) throws Exception {
        System.out.println("The client receive response of http header is : " + msg.getHttpResponse().headers().names());
        System.out.println("The client receive response of http body is : " + msg.getResult());
    }
}
