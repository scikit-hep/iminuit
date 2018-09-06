This was synced with the Minuit2 source from ROOT using the standalone build.

To re-sync:

Assuming you have checked out ROOT to `~/ROOT` and iminuit to `~/iminuit`, run the following commands (delete the contents of this folder first if you want to make sure to account for deleted files - but don't delete this readme):

```bash
cd ~/ROOT/math/minuit2
mkdir build
cd build
cmake .. -Dminuit2_standalone=ON
cp -r  ../src ../inc ../LICENSE ../LGPL2_1.txt ../version_number ~/iminuit/Minuit/
```

You now have a fresh copy of Minuit2!
