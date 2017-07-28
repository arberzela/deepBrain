function epoch = getchannel(header, cn, t1,t2)
% function epoch = getchannel(header, cn, t1, t2)
% header = header structure array
% cn = channel number
% t1 = erster samplepunkt, der gelesen werden soll
% t2 = letzter samplepunkt, der gelesen werden soll
%
% zum einlesen z.B. der ersten 2048 Datenpunkte:
% a=getchannel(file,cn,1,2048);

samplepoints=t2-t1+1;
epoch = zeros(samplepoints,1);
fid = fopen([header.source,'.data.bin'],'r');
if t1>1 | cn>1
    to_skip = (cn-1)*header.nop*4+(4*t1-8);
    skip_samples=fread(fid,1,'single',to_skip);
end
epoch=fread(fid,samplepoints,'single');
fclose(fid); 