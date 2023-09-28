import xml.etree.ElementTree as et 
import pandas as pd
import os

def dataframeSetup():
    total_stat_cols = ["gameID", "teamID", "total_rush", "total_pass", "total_punt", "total_ko", "total_pat"]
    total_stats = pd.DataFrame(columns=total_stat_cols)
    total_stats.set_index(total_stats["gameID"],inplace = True)

    play_df_cols = ["playID", "type", "clock", "token", "context", "text"]
    play_df = pd.DataFrame(columns=play_df_cols)
    play_df.set_index(play_df["playID"], inplace=True)
    return total_stats, play_df

def getFilenameList(dir):
    rtn = []
    for root,dirs,files in os.walk(dir):
        for name in files:
            filepath = root + os.sep + name
            if filepath.endswith(".xml"):
                rtn.append(filepath)
                #print(filepath)
                #xml_parser(filepath)
                #rtn.append(os.path.join(root,name))
    return rtn


def xml_parser(fileNames):
    total_stats, play_df = dataframeSetup()
    
    for fileName in fileNames:    
        xmlFile = fileName
        xtree = et.parse(xmlFile)
        xroot = xtree.getroot()
        for node in xroot:
            if node.tag == "venue":
                gameID = node.attrib['gameid']
                print(gameID)
            elif node.tag == "team":
                teamID = node.attrib['id']
                totals = node.find("totals")
                try:
                    total_rush = totals.find("rush").attrib['att']
                except AttributeError:
                    total_rush = 0
                try:
                    total_pass = totals.find("pass").attrib['att']
                except AttributeError:
                    total_pass = 0
                try:
                    total_punt = totals.find("punt").attrib['no']
                except AttributeError:
                    total_punt = 0
                try:
                    total_ko = totals.find("ko").attrib['no']
                except AttributeError:
                    total_ko = 0
                try:    
                    total_pat = totals.find("pat").attrib['kickatt']
                except KeyError:
                    total_pat = 0
                except AttributeError:
                    total_pat = 0
                
                total_stats = total_stats.append({"gameID": gameID, "teamID": teamID, "total_rush":total_rush, 
                "total_pass": total_pass, "total_punt": total_punt, "total_ko": total_ko, "total_pat": total_pat}, ignore_index=True)
            elif node.tag == "plays":

                for quarter in node:
                    plays = quarter.findall("play")
        #             <play context="V,2,9,V03" playid="15,5,151" type="P" pcode="MI" first="P" tokens="PASS:12,C,11,V23,MI TACK:3"
        #   text="Lombardi,Rocky middle pass complete to Travis,Messiah for 20 yards to the HUSKIES23, 1ST DOWN HUSKIES (Swilling, T.)."></play>
                    for play in plays:
                        if play.tag == "play":
                            playContext = play.attrib.get('context')
                            playID = play.attrib.get('playid')
                            playType = play.attrib.get('type')
                            clock = play.attrib.get('clock')
                            playToken = play.attrib.get('tokens')
                            playText = play.attrib.get('text')
                            play_df = play_df.append({"playID":playID, "type":playType, "clock":clock, "token":playToken, "context":playContext, "text":playText}, ignore_index = True)

    play_df.to_csv('allPlays.csv')
    total_stats.to_csv('totalStats.csv')
    return total_stats, play_df
# xml_parser(["GT090821.xml"])
# print(getFilenameList("."))
# xml_parser(getFilenameList("."))


