newoption {
    trigger = "no-crt",
    description = "Disable linking with the Visual C++ Runtime Library on Windows."
}

newoption {
    trigger = "python",
    value = "path",
    description = "Path to Python 3.7 (e.g. C:/ProgramData/Anaconda3/, we use C:/ProgramData/Anaconda3/include etc...)"
}

workspace "DataFlow"
    architecture "x64"
    configurations { "Debug", "Release", "Dist" }

function common_settings()
    architecture "x64"

    language "C++"

    -- We can't specify C++20 but at least on Windows, our generate_projects.bat replaces language standard with stdcpplatest in the .vcxproj files
    cppdialect "C++17" 

    rtti "Off"
    characterset "Unicode"
    
    editandcontinue "Off"

    defines "_HAS_EXCEPTIONS=0"
    exceptionhandling "Off"
	
    includedirs { "%{prj.name}/src" }

    filter "system:windows"
        excludes "%{prj.name}/src/posix_*.cpp"
        systemversion "latest"
        buildoptions { "/utf-8" }
        
        defines { "NOMINMAX", "WIN32_LEAN_AND_MEAN", "_CRT_SECURE_NO_WARNINGS" }
        links { "dwmapi.lib", "dbghelp.lib" }

    -- Exclude windows files on non-windows platforms since they would cause a compilation failure
    filter { "system:windows", "not options:no-crt" }
        staticruntime "On"
        excludes "%{prj.name}/src/windows_no_crt.cpp"

    filter { "system:windows", "options:no-crt" }
        defines "BUILD_NO_CRT"
        flags { "NoRuntimeChecks", "NoBufferSecurityCheck" }
        buildoptions { "/Gs9999999" }
    filter { "system:windows", "options:no-crt", "not kind:StaticLib" }
        linkoptions { "/nodefaultlib", "/subsystem:windows", "/stack:\"0x100000\",\"0x100000\"" }
        links { "kernel32", "shell32", "winmm", "ole32" }
        flags { "OmitDefaultLibrary" }
    filter { "system:windows", "options:no-crt", "kind:SharedLib" }
        entrypoint "main_no_crt_dll"
    filter { "system:windows", "options:no-crt", "kind:ConsoleApp or WindowedApp" }
        entrypoint "main_no_crt"

    filter "configurations:Debug"
        defines "DEBUG"
        symbols "On"
        buildoptions { "/FS" }
    filter "configurations:Release"
        defines "RELEASE"
        optimize "On"
        symbols "On"
        buildoptions { "/FS" }
    filter "configurations:Dist"
        defines "DIST"
        optimize "Full"
        floatingpoint "Fast"
    filter {}
end


outputFolder = "%{cfg.buildcfg}-%{cfg.system}-%{cfg.architecture}"

project "lstd"
    location "%{prj.name}"
    kind "StaticLib"

    targetdir("bin/" .. outputFolder .. "/%{prj.name}")
    objdir("bin-int/" .. outputFolder .. "/%{prj.name}")

    files {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.inc",
        "%{prj.name}/src/**.c",
        "%{prj.name}/src/**.cpp",
        "%{prj.name}/src/**.def",
		"%{prj.name}/src/**.ixx"
    }

    filter {}

    pchheader "pch.h"
    pchsource "%{prj.name}/src/pch.cpp"
    forceincludes { "pch.h" }
    
    common_settings()

project "main"
    location "%{prj.name}"
    kind "ConsoleApp"

    targetdir("bin/" .. outputFolder .. "/%{prj.name}")
    objdir("bin-int/" .. outputFolder .. "/%{prj.name}")

    files {
        "%{prj.name}/src/**.h",
        "%{prj.name}/src/**.cpp"
    }

    links { "lstd" }
    includedirs { "lstd/src" }

    pchheader "pch.h"
    pchsource "%{prj.name}/src/pch.cpp"
    forceincludes { "pch.h" }
    
    common_settings()
