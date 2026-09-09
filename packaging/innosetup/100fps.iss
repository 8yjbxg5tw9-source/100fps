; 100fps — Windows installer (Step 10, Inno Setup 6).
;
; Per-user install (no admin rights needed): the app lands in
; %LOCALAPPDATA%\100fps, which is always writable, so models/, workspace/
; and the auto-updating weights cache work out of the box.
;
; Build steps (on Windows):
;   1. python packaging\build.py --weights download   (portable -> dist\100fps)
;   2. Compile this script with Inno Setup 6 (iscc 100fps.iss)
;   3. Setup lands in packaging\dist\installer\100fps-<ver>-win64-setup.exe
;
; NOTE: MyAppVersion MUST match pipeline.__version__ (enforced by
; tests/test_step10_packaging.py::test_iss_version_matches_package).

#define MyAppName "100fps"
#define MyAppVersion "1.0.0"
#define MyAppPublisher "100fps"
#define MyAppURL "https://github.com/8yjbxg5tw9-source/100fps"
#define MyAppExeName "100fps-gui.exe"
#define MyAppCliName "100fps.exe"

[Setup]
AppId={{3237A1A9-8268-4140-818A-D75E77650884}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\dist\installer
OutputBaseFilename=100fps-{#MyAppVersion}-win64-setup
SetupIconFile=..\assets\icon.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0
; ~4 GB installed (torch CUDA + GUI), ~130 MB more if weights pre-bundled.
ExtraDiskSpaceRequired=6442450944
UninstallDisplayName={#MyAppName} {#MyAppVersion} (720p -> 8K @ 1000 FPS)

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "cliicon"; Description: "Start Menu shortcut for the CLI (100fps.exe)"; \
    GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Portable dir produced by packaging/build.py (onedir target).
Source: "..\dist\100fps\*"; DestDir: "{app}"; Flags: recursesubdirs \
    createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Comment: "100fps — 720p video to 8K @ 1000 FPS (GUI)"
Name: "{group}\{#MyAppName} CLI"; Filename: "{app}\{#MyAppCliName}"; \
    Comment: "100fps command line"; Tasks: cliicon
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Comment: "100fps — 720p video to 8K @ 1000 FPS (GUI)"; Tasks: desktopicon
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Keep user data (models, workspaces) — only remove install-time caches.
Type: filesandordirs; Name: "{app}\__pycache__"

[Code]
function HasNvidiaDriver(): Boolean;
var
  Smi: String;
begin
  { nvidia-smi.exe ships with every NVIDIA driver on 64-bit Windows. }
  Smi := ExpandConstant('{sys}\nvidia-smi.exe');
  Result := FileExists(Smi);
end;

function InitializeSetup(): Boolean;
var
  Answer: Integer;
begin
  Result := True;
  if not HasNvidiaDriver() then
  begin
    Answer := MsgBox(
      'No NVIDIA GPU driver was detected (nvidia-smi.exe not found).' + #13#10 +
      #13#10 +
      '100fps will still install and run in CPU mode, but 8K @ 1000 FPS ' +
      'on CPU is EXTREMELY slow (hours per second of video).' + #13#10 +
      #13#10 +
      'Recommended: install the latest NVIDIA Studio / Game Ready driver ' +
      'for your GPU, then continue.' + #13#10 +
      #13#10 +
      'Continue with the installation anyway?',
      mbConfirmation, MB_YESNO);
    if Answer = IDNO then
      Result := False;
  end;
end;

function InitializeUninstall(): Boolean;
begin
  Result := MsgBox(
    'Remove 100fps?' + #13#10 + #13#10 +
    'Your downloaded AI models (models\) and workspaces are KEPT — ' +
    'delete the install folder manually if you want them gone too.',
    mbConfirmation, MB_YESNO) = IDYES;
end;
