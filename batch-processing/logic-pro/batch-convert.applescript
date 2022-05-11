activate application "Logic Pro X"

tell application "Logic Pro X"
	open file "Macintosh HD:Users:stefanh:Documents:UMich:Research:Music with NLP:datasets:LMD-cleaned_broken:Nirvana - Been a Son.mid"
end tell

#tell application "System Events" to tell process "Logic Pro X"
#	click menu item "Open Score Editor" of menu 1 of menu bar item "Window" of menu bar 1
#end tell

tell application "System Events" to tell process "Logic Pro X"
	click menu item "Open Score Editor" of menu 1 of menu bar item "Window" of menu bar 1
	delay 0.2
	
	click menu item "Score as MusicXMLÉ" of menu 1 of menu item "Export" of menu 1 of menu bar item "File" of menu bar 1
	delay 0.3
	
	tell window "Save MusicXML File as:"
		keystroke "test"
		delay 0.2
		
		keystroke tab # open `go to`
		key down shift
		key down command
		keystroke "g"
		key up shift
		key up command
		delay 0.3
		
		set filesavepath to "/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_broken"
		keystroke filesavepath
		delay 0.3
		
		keystroke return
		delay 0.2
		
		click button "Save"
	end tell
	
	perform action "AXRaise" of window 1 # bring Score Editor window to front	
	key down command # close Score window
	keystroke "w"
	key up command
	
	click menu item "Close" of menu 1 of menu bar item "File" of menu bar 1 # close current file
	keystroke space
end tell

