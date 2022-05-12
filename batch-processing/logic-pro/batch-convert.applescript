# Batch processing a folder of midi files 
# intended for a folder containing midi files only
set dir_process to "/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_broken, test/"

# taken from https://stackoverflow.com/a/12535220/10732321
on remove_extension(this_name)
	if this_name contains "." then
		set this_name to Â
			(the reverse of every character of this_name) as string
		set x to the offset of "." in this_name
		set this_name to (text (x + 1) thru -1 of this_name)
		set this_name to (the reverse of every character of this_name) as string
	end if
	return this_name
end remove_extension

set dir_process_fl to POSIX file dir_process
tell application "Finder"
	set songs to files of folder dir_process_fl as alias list
end tell

activate application "Logic Pro X"

repeat with f in songs
	log f
	tell application "Logic Pro X"
		open file f
	end tell
	
	set fnm to name of (info for f)
	set fnm to remove_extension(fnm)
	
	tell application "System Events" to tell process "Logic Pro X"
		click menu item "Open Score Editor" of menu 1 of menu bar item "Window" of menu bar 1
		delay 0.2
		
		click menu item "Score as MusicXMLÉ" of menu 1 of menu item "Export" of menu 1 of menu bar item "File" of menu bar 1
		delay 0.6 # sometimes the dialog takes longer to load
		
		tell window "Save MusicXML File as:"
			keystroke fnm
			delay 0.3
			
			keystroke tab # open `go to`
			key down shift
			key down command
			keystroke "g"
			key up shift
			key up command
			delay 0.3
			
			keystroke dir_process
			delay 0.4
			
			keystroke return
			click button "Save"
			delay 0.4 # delay needed as some songs are long
		end tell
		
		perform action "AXRaise" of window 1 # bring Score Editor window to front	
		key down command # close Score window
		keystroke "w"
		key up command
		
		click menu item "Close" of menu 1 of menu bar item "File" of menu bar 1 # close current file
		keystroke space
	end tell
end repeat


