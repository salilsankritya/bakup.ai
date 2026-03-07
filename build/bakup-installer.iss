; ─────────────────────────────────────────────────────────────────────────────
; bakup.ai — Inno Setup Installer Script
; ─────────────────────────────────────────────────────────────────────────────
; Generates: bakup-ai-installer.exe
;
; Usage:
;   Install Inno Setup 6 (https://jrsoftware.org/isdl.php)
;   Open this file in Inno Setup Compiler, or run:
;     iscc build\bakup-installer.iss
; ─────────────────────────────────────────────────────────────────────────────

#define MyAppName "bakup.ai"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "bakup.ai"
#define MyAppURL "https://github.com/salilsankritya/bakup.ai"
#define MyAppExeName "bakup-launcher.bat"

[Setup]
AppId={{B4KUP-A1-D3V-PR3V13W-2026}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
DefaultDirName={autopf}\bakup-ai
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=..\dist
OutputBaseFilename=bakup-ai-installer
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesInstallIn64BitMode=x64compatible
SetupLogging=yes
UninstallDisplayName={#MyAppName}

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; Compiled backend binary + all dependencies
Source: "..\dist\bakup-server\*"; DestDir: "{app}\bakup-server"; Flags: ignoreversion recursesubdirs createallsubdirs

; Launcher script
Source: "bakup-launcher.bat"; DestDir: "{app}"; Flags: ignoreversion

[Dirs]
; Create runtime data directories
Name: "{app}\data"
Name: "{app}\data\vectordb"
Name: "{app}\data\model-weights"

[Icons]
; Start menu shortcut
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Comment: "Start bakup.ai local server"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
; Desktop shortcut (optional task)
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; Tasks: desktopicon; Comment: "Start bakup.ai local server"

[Run]
; Launch app after installation
Filename: "{app}\{#MyAppExeName}"; Description: "Launch {#MyAppName}"; Flags: nowait postinstall skipifsilent shellexec

[UninstallDelete]
; Clean up runtime data on uninstall (optional — user data)
Type: filesandordirs; Name: "{app}\data"

[Code]
// Kill any running bakup-server.exe before uninstalling
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  ResultCode: Integer;
begin
  if CurUninstallStep = usUninstall then
  begin
    Exec('taskkill.exe', '/F /IM bakup-server.exe', '', SW_HIDE, ewWaitUntilTerminated, ResultCode);
  end;
end;
