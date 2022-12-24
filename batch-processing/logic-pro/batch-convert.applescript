# Batch processing a folder of midi files
# intended for a folder containing midi files only
# set dir_process to "/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/converted/LMD-cleaned, LP, todo/"
# set dir_process to "/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/POP909, LP, todo/"
# set dir_process to "/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/MAESTRO, todo/"
# set dir_process to "/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD/00000, todo/"
# set dir_process to "/Users/stefanh/Documents/UMich/Research/Music with NLP/datasets/LMD-cleaned_broken, todo/"
# set dir_process to "/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/converted/LMD, LP/140000-150000, todo"
set dir_process to "/Users/stefanhg/Documents/UMich/Research/Music with NLP/datasets/converted/LMCI, LP/010000-020000, todo/"


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
		activate application "Logic Pro X"
		open file f
	end tell
	
	set fnm to name of (info for f)
	set fnm to remove_extension(fnm)
	
	tell application "System Events" to tell process "Logic Pro X"
		click menu item "Open Score Editor" of menu 1 of menu bar item "Window" of menu bar 1
		delay 0.2
		
		click menu item "Score as MusicXMLÉ" of menu 1 of menu item "Export" of menu 1 of menu bar item "File" of menu bar 1
		delay 0.5 # sometimes the dialog takes longer to load
		
		tell window "Save MusicXML File as:"
			delay 0.1
			#			keystroke fnm
			set the clipboard to fnm # cos keystroke doesn't work with Chinese characters 
			delay 0.2
			key down command # pasting is also faster than typing
			keystroke "v"
			key up command
			delay 0.3
			
			# once path set once, becomes default path for next save
			keystroke tab # open `go to` to set the save path
			#key down shift
			#key down command
			#keystroke "g"
			#key up shift
			#key up command
			#delay 0.3
			
			#			keystroke dir_process
			#set the clipboard to dir_process
			#key down command
			#keystroke "v"
			#key up command
			#delay 0.3
			
			#keystroke return
			click button "Save"
		end tell
		delay 0.8 # delay needed as some songs are long
		
		perform action "AXRaise" of window 1 # bring Score Editor window to front
		delay 0.2
		key down command # close Score window
		keystroke "w"
		key up command
		
		click menu item "Close Project" of menu 1 of menu bar item "File" of menu bar 1 # close current file		
		delay 0.2
		click button "DonÕt Save" of window 1
		# keystroke space
	end tell
end repeat


