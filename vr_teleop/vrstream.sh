docker run --rm -it --network=host \
  --add-host xr.kscale.ai:10.33.20.104 \
  --add-host host.docker.internal:127.0.0.1 \
  webrtc:1.24.9-datac gst-launch-1.0 \
  webrtcsink name=ws meta="meta,name=kbotv2-rpi4b" enable-control-data-channel=true signaller::uri="wss://xr.kscale.ai:8585" \
  compositor name=mix sink_0::xpos=0 sink_1::xpos=1280 ! videoconvert ! video/x-raw,format=I420 ! ws. \
  videotestsrc pattern=0 ! video/x-raw,format=YUY2,width=1280,height=1080 ! videoconvert ! videoflip method=vertical-flip ! video/x-raw,format=I420 ! mix.sink_0 \
  videotestsrc pattern=1 ! video/x-raw,format=YUY2,width=1280,height=1080 ! videoconvert ! videoflip method=vertical-flip ! video/x-raw,format=I420 ! mix.sink_1
