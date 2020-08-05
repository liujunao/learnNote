package com.phei.netty.protocol.twelve.server;

import io.netty.channel.ChannelHandlerAdapter;
import io.netty.channel.ChannelHandlerContext;

import com.phei.netty.protocol.twelve.MessageType;
import com.phei.netty.protocol.twelve.struct.Header;
import com.phei.netty.protocol.twelve.struct.NettyMessage;
import org.apache.commons.logging.Log;
import org.apache.commons.logging.LogFactory;

public class HeartBeatRespHandler extends ChannelHandlerAdapter {
    private static final Log LOG = LogFactory.getLog(HeartBeatRespHandler.class);

    @Override
    public void channelRead(ChannelHandlerContext ctx, Object msg) throws Exception {
        NettyMessage message = (NettyMessage) msg;
        // 返回心跳应答消息
        if (message.getHeader() != null && message.getHeader().getType() == MessageType.HEARTBEAT_REQ.value()) {
            LOG.info("Receive client heart beat message : ---> " + message);
            NettyMessage heartBeat = buildHeatBeat();
            LOG.info("Send heart beat response message to client : ---> " + heartBeat);
            ctx.writeAndFlush(heartBeat);
        } else
            ctx.fireChannelRead(msg);
    }

    private NettyMessage buildHeatBeat() {
        NettyMessage message = new NettyMessage();
        Header header = new Header();
        header.setType(MessageType.HEARTBEAT_RESP.value());
        message.setHeader(header);
        return message;
    }
}
