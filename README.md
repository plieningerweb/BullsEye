# BullsEye
Bull's Eye - Never miss a shot

Idea: Track a dart in realtime and move bullseye to the calculated target of the dart. Therefore dart will always hit the center!

##### Hardware: DVS / eDVS

Two dynamic vision sensors (DVS) or two embedded dynamic vision sensors (eDVS) connected with a cable to sync microtimestamp


## How to run the example?

```
python2 ./python-scripts/edvs-tracker-3D.py
```

## How to record data and evaluate shot?

1. Install jAER using SVN checkout from https://svn.code.sf.net/p/jaer/code/jAER/trunk/
2. Run trunk/jAER/jAER.sh
3. Connect two DVS Cameras to your computer
4. Maybe setup USB on Linux to recognize cameras (check https://sourceforge.net/p/jaer/wiki/)
5. Connect the two DVS Cameras usinga  sync cable (check http://siliconretina.ini.uzh.ch/wiki/doku.php?id=userguide#dvs128_camera_synchronization): No need to connect GND, because USB is on same computer
```
Quote form: http://siliconretina.ini.uzh.ch/wiki/doku.php?id=userguide#dvs128_camera_synchronization)
DVS128 Camera synchronization

Multiple cameras can be synchronized to microsecond time-stamp precision. To connect cameras, you can use coax cables together with t-connectors and the Pomona Electronics 5069 coax breakout adaptor with 0.025” pin connectors (e.g. http://www.digikey.com/product-search/en?x=-1075&y=-87&lang=en&site=us&KeyWords=pomona+5069 or http://www.testpath.com/Items/Breakout-BNC-Female-to-0025-in-Square-Pin-Sockets-113-531.htm).

The DVS128 cameras can be precisely synchronized so that simultaneous events from two cameras will receive exactly the same timestamps. The camera have the capability that one camera can be the master timestamp source which clocks the the other's timestamp counter.

For synchronization, DVS128 cameras with a firmware version number >= 11 should be used, older firmware versions are not synchronized properly if the USB load is high. To enable this functionality, the cameras must be connected so that the OUT of the master is connected to the IN of the slave(s). If the cameras are connected to the same computer, the ground pins do not have to be connected (to avoid possible ground loops). If the cameras are connected to different computers, the ground pins should be connected as shown below.

Starting from firmware version 11, the timestamp master camera has to be selected in software. In the jAER software, for the timestamp master (where the OUT is connected) the checkbox 'Timestamp master / enable sync event output' in the DVS128-menu has to be checked, for the slave(s), this checkbox has to be unchecked.

The master camera keeps its bottom LED lit, while the slaves extinguish theirs.

To synchronize the cameras after plugging them in, press '0' in the AEViewer window of the timestamp master to reset the timestamps of all the cameras.
```
(next steps continue here:)

6. open jAER Instance, choose one Camera from the menu
7. open a new jAER viewer, form menu -> new viewer
8. select second camera in new viewer
9. set synchronization in master camera to master
10. uncheck "master" on slave camera menu
11. go to master window and press "0" (zero) to reset the sync timer: Now the status field on the bottom of the screen should say, that the timer was successfully reset
12. Start recording on both windows
13. Throw a dart so both cameras can see it (camera should be setup like described in the section below)
14. Stop recording and store files
15. Edit ./python-scripts/edvs-tracker-3D.py and change `datafile = 'pathtofilename1.aedat' and datfile2 = 'pathtofilename2.aedat'
16. run `run python2 ./python-scripts/edvs-tracker-3D.py`
17. to pause visualisation, click in image, to continue click again
18. you should finally see the estimated flight curve and target

## Setup

We used this sketch to setup our cameras. In an advanced version the software shoudl autocalibrate the cameras. 

## Concolusions and Outlook

The shown example can track the dart, but is still not accurate enough. Additionally, it seems to be possible to track the dart fast enough for realtime shots. This project can be used as basis to develop an advanced realtime tracking mechanism including software calibration, e.g. on an embedded platfrom.

## Further Information

This project was part of a class work at Technical University of Munich (TUM), and in the subfolder `presentations` you can find further information)
