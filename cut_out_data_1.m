%% PART 1: absolute spektrale Magnitude f�r jede der markierten Tags separat rechnen

clear
%path(path,genpath('/usr/share/matlab')) % das Matlab-Verzeichnis mit allen Funktionen bestimmen

%% WAS ANGEGEBEN WERDEN MUSS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Patient = {'P1'}; % !!!!! FUR WELCHEN PATIENT SOLL DIE RECHNUNG GEMACHT WERDEN?

compute_channelwise_ps = 1; % Werte 1 oder 0 in Abh�ngigkeit davon
% ob man absolute Spektren pro Header pro Kanal f�r einzelne Tags rechnen und im Ordner 
% F:\Rajbir\Analysen\'Pat'\Spektren\'Tag'\ speichern m�chte (1 angeben)
% oder ob diese schon gerechnet worden sind (0). Der wert soll Z.B., "1":
% sein, wenn Du Sperkten f�r den neuen Marker "lonsn" rechnest. Zeile 32
% soll in diesem Fall nicht kommentiert sein, und Zeilen 27-30 sollen
% wegkommentiert werden.

compute_spectra = 1; % Werte 1 oder 0 in Abh�ngigkeit davon
% ob man Kanal- und Header-gemittelte Spetktren aus den ps-en, welche in "compute_channelwise_ps" 
% erstellt worden sind, f�r einzelne Tags als eine Variable im Ordner
% F:\Rajbir\Analysen\'Pat'\Spektren erstellen m�chte (1) oder ob alle
% dieser Variablen schon vorhanden sind (0). "1" soll zur Sicherheit immer angegeben
% werden.

alle_Stimuli = 'ps';
% alle_Stimuli = ['ss ', 'ps ', 'afs ', 'as ', 'ns ', 'vs ', 'pas ', 'pwavs'];
%alle_Stimuli = ['nons ', 'nonsm ', 'lonspp ', 'lonsppm ', 'lonsnpl ', 'lonsnplm ', 'lonspnl ', 'lonspnlm ','lonsnnll ', 'lonsnnllm ', 'lonsnnhh ', 'lonsnnhhm ','lonsnnhl ', 'lonsnnhlm ','lonsnnlh']; %  F�r lons + Ger�usche nach dem onset
idx = []
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% alle_Stimuli=strsplit(alle_Stimuli, ' ');

kanaele=122; % all channels
Path = ('C:\Users\user\Desktop\Master Project\data\CAR_EEG\P1\');  % Path where the Header-files are saved
Save_all_data = ('C:\Users\user\Desktop\Master Project\data\CAR_EEG\P1\'); % wo es gespeichert werden soll


%% F�r jeden Tag separat werden Spektren gerechnet
if compute_spectra
  %  for yy = 1:length(alle_Stimuli)
        
        Stimulus = alle_Stimuli; % nimm den yy-ten Stimulus
        
        Savepath= (['C:\Users\user\Desktop\Master Project\data\CAR_EEG\P1\asm' char(Stimulus) '\']);
        
        %% Header identifizieren, welch den Stimulus beinhalten
        cd(Path)
        D = dir([Path '*.hdr.mat']);
        Hmat= {D.name}';
        for i = 1 : size(Hmat,1) % f�r jeden header
            load([Path Hmat{i}]);
            [a] = strmatch(Stimulus,H.itmxtickl','exact'); % suche : 1) was, 2) wo? 3) wie genau?
            if size(a,1)>0,
                idx(i) = 1;
            else
                idx(i) = 0;
            end
            clear a
        end
        
        Hliste = {Hmat{[find(idx)]}};  % List of all Header-files where the Stimulus is present
        
        %% Spektren berechnen:
        if ~isempty(Hliste)  % wenn die Liste aus Headern mit markierungen 1 oder mehr Header beinhaltet
            if compute_channelwise_ps
                
                for r=1:numel(Hliste)
                    
                    disp(['r=' num2str(r)])
                    
                    load(char(Hliste{r}));                         %load header
                    %% Parameter f�r die spektrale Analyse
%                       ZF=round(H.sf/2);              % T - length of time window
%                       Schritt=round(ZF/10);            % tstep - time steps in which time window is moved    
%                     ZF=round(H.sf/2);              % T - length of time window
%                     Schritt=round(ZF/10);            % tstep - time steps in which time window is moved                    
%                     ZF=round(H.sf/20);              % T - length of time window
%                     Schritt=round(ZF/10);            % tstep - time steps in which time window is moved
                    a = strmatch(Stimulus,H.itmxtickl','exact');
                    pe = strmatch('pe',H.itmxtickl','exact');
                    
                    clear ps
                    T=[2048 6144];                                %2Sekunden in Samplepunkten vor- und nach dem Marker
                    
                    data=zeros(kanaele,sum(T)+1,length(a));         %zero-Matrix with dimensions  channels x time x trials
                    diff = zeros(length(pe),1);
                    channels = cell(length(kanaele),1)
                    %% die leere Matrix "data" mit Daten f�llen
                    c=1;                                          %counter
                    for i=a' % for all Markers defined before, der Vektoer wird um 90 Grad gedreht
                        index = find(a==i);
                        peEl = pe(index);
                        disp(['...reading trial ',num2str(c)])    % displays current trial
                        t=H.itmxtick(i);   % searches time point corresponding to the markers
                        diff(index,1) = H.itmxtick(peEl) -t ;
                        if t>T(1) && t<(H.nop-T(2))               % checks that maker is not to close to beginning/end of data
                            for ch=1:kanaele   % for all channels
                                channels{ch} = H.cn{ch};
                                data(ch,1:sum(T)+1,c)=getchannel(H,ch,t-T(1),t+T(2));   % cuts out time window around marker for all channels (data: channel*time*trials)
                           end
                        end
                        c=c+1;                                    % counts to the next marker
                        
                    end
%                     %% Absolute Spektren rechnen
%                     for ch=1:kanaele                                              % for all channels
%                         disp(['...processing channel ',num2str(ch)])             %zeigt an, welcher Kanal berechnet wird
%                         
%                         data2=squeeze(data(ch,:,:));                             %squeeze: cuts Dimensionen, which are 1 (Berechnung geht dann schneller)
%                         
%                         if size(data2,1)==1                                      %important for only one trial per header, because 'data' shifts in these cases
%                             data2=data2';
%                         end
%                         name = char (Hliste(r));
%                         %[ps,psc,frq,ts]=tfpsmtapernew_impoved(data2,ZF,Schritt,H.sf,1.5,1,1,0);   % multitaper (ps: frequency bins x time bins x Trials)
%                         [ps,psc,frq,ts]=tfpsmtapernew_impoved(data2,ZF,Schritt,H.sf,3,1,0,0);   % multitaper (ps: frequency bins x time bins x Trials)
%                         save([Savepath, char(Stimulus), num2str(ch),'_asm',name(1:13),'.mat'],'ps','frq','ts');  % save ps
%                     end
%                 end
%             end
%             
%             %% combine all trials
%             clear allps allallps
%             
%             for ch=1:kanaele;                         % for all channels
%                 tr=0;
%                 for r=1:length(Hliste);             % for all elements of the header list
%                     name = char (Hliste(r));
%                     load ([Savepath, char(Stimulus), num2str(ch),'_asm',name(1:13),'.mat']);
%                     allps(:,:,(tr+1):(tr+size(ps,3)))=ps;         % all trials are cut together, resulting in one ps-file
%                     tr=tr+size(ps,3);
%                 end
%                 
%                 ps=allps;
%                 allallps(ch,:,:,:)=ps;
%                 
%             end
            
            cd(Save_all_data)
            disp(['...saving ' char(Stimulus), '.mat'])
            %Filename = strcat(string(Stimulus),string(r));
            save([char(Stimulus), num2str(r), '.mat'],'data','diff','channels','-v7.3');
           
                end
        
    end
end
%    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
